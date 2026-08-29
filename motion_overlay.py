from collections import deque
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np

from motion_analysis import (
    POSE_MODEL_PATH,
    HAND_MODEL_PATH,
    POSE_INDEX,
    calculate_angle_2d,
    get_phase,
    get_video_info,
)


# =========================================================
# OT Practice Copilot
# 動画オーバーレイ解析
# =========================================================


# =========================================================
# Poseの骨格接続
# =========================================================

POSE_CONNECTIONS = [
    # 上肢
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),

    # 肩
    (11, 12),

    # 体幹
    (11, 23),
    (12, 24),
    (23, 24),

    # 左下肢
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),

    # 右下肢
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
]


# =========================================================
# Handの骨格接続
# =========================================================

HAND_CONNECTIONS = [
    # 母指
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # 示指
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # 中指
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # 環指
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # 小指
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # 手掌
    (0, 17),
]


# =========================================================
# 座標変換
# =========================================================

def landmark_to_pixel(
    landmark,
    width,
    height,
):
    """
    MediaPipeの正規化座標を
    動画上のピクセル座標へ変換する。
    """

    x = int(
        landmark.x * width
    )

    y = int(
        landmark.y * height
    )

    return (
        x,
        y,
    )


# =========================================================
# ランドマークが描画可能か確認
# =========================================================

def is_landmark_visible(
    landmark,
    threshold=0.4,
):
    """
    Poseランドマークのvisibilityを確認する。
    Handランドマークにはvisibilityがないため、
    その場合はTrueを返す。
    """

    visibility = getattr(
        landmark,
        "visibility",
        None,
    )

    if visibility is None:
        return True

    return (
        visibility >= threshold
    )


# =========================================================
# Pose骨格描画
# =========================================================

def draw_pose_skeleton(
    frame,
    landmarks,
):
    """
    全身の関節点と骨格線を動画へ描画する。
    """

    height, width = (
        frame.shape[:2]
    )

    left_color = (
        80,
        220,
        80,
    )

    right_color = (
        80,
        170,
        255,
    )

    line_color = (
        220,
        220,
        220,
    )


    # -----------------------------------------------------
    # 骨格線
    # -----------------------------------------------------

    for start_index, end_index in (
        POSE_CONNECTIONS
    ):

        start_landmark = (
            landmarks[
                start_index
            ]
        )

        end_landmark = (
            landmarks[
                end_index
            ]
        )

        if not (
            is_landmark_visible(
                start_landmark
            )
            and
            is_landmark_visible(
                end_landmark
            )
        ):
            continue

        start_point = (
            landmark_to_pixel(
                start_landmark,
                width,
                height,
            )
        )

        end_point = (
            landmark_to_pixel(
                end_landmark,
                width,
                height,
            )
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            line_color,
            2,
            cv2.LINE_AA,
        )


    # -----------------------------------------------------
    # 関節点
    # -----------------------------------------------------

    left_indices = {
        11,
        13,
        15,
        23,
        25,
        27,
        29,
        31,
    }

    right_indices = {
        12,
        14,
        16,
        24,
        26,
        28,
        30,
        32,
    }


    for index in (
        left_indices
        | right_indices
    ):

        landmark = (
            landmarks[index]
        )

        if not is_landmark_visible(
            landmark
        ):
            continue

        point = landmark_to_pixel(
            landmark,
            width,
            height,
        )

        color = (
            left_color
            if index in left_indices
            else right_color
        )

        cv2.circle(
            frame,
            point,
            6,
            color,
            -1,
            cv2.LINE_AA,
        )


# =========================================================
# 関節角度描画
# =========================================================

def draw_angle(
    frame,
    landmarks,
    point_a_index,
    joint_index,
    point_c_index,
    label,
):
    """
    A-B-Cで構成される2D投影角を
    関節の近くに表示する。
    """

    height, width = (
        frame.shape[:2]
    )

    a = landmarks[
        point_a_index
    ]

    b = landmarks[
        joint_index
    ]

    c = landmarks[
        point_c_index
    ]


    if not (
        is_landmark_visible(a)
        and
        is_landmark_visible(b)
        and
        is_landmark_visible(c)
    ):

        return


    angle = calculate_angle_2d(
        a,
        b,
        c,
    )


    if np.isnan(
        angle
    ):
        return


    x, y = landmark_to_pixel(
        b,
        width,
        height,
    )


    text = (
        f"{label} "
        f"{angle:.0f}deg"
    )


    # 黒い縁取り
    cv2.putText(
        frame,
        text,
        (
            x + 8,
            y - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            0,
            0,
            0,
        ),
        3,
        cv2.LINE_AA,
    )


    # 白文字
    cv2.putText(
        frame,
        text,
        (
            x + 8,
            y - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )


# =========================================================
# 全身角度描画
# =========================================================

def draw_pose_angles(
    frame,
    landmarks,
):
    """
    左右の主要関節について
    2D投影角を動画上に表示する。
    """

    p = POSE_INDEX


    # -----------------------------------------------------
    # 左
    # -----------------------------------------------------

    draw_angle(
        frame,
        landmarks,
        p["left_hip"],
        p["left_shoulder"],
        p["left_elbow"],
        "L Shoulder",
    )

    draw_angle(
        frame,
        landmarks,
        p["left_shoulder"],
        p["left_elbow"],
        p["left_wrist"],
        "L Elbow",
    )

    draw_angle(
        frame,
        landmarks,
        p["left_shoulder"],
        p["left_hip"],
        p["left_knee"],
        "L Hip",
    )

    draw_angle(
        frame,
        landmarks,
        p["left_hip"],
        p["left_knee"],
        p["left_ankle"],
        "L Knee",
    )

    draw_angle(
        frame,
        landmarks,
        p["left_knee"],
        p["left_ankle"],
        p["left_foot_index"],
        "L Ankle",
    )


    # -----------------------------------------------------
    # 右
    # -----------------------------------------------------

    draw_angle(
        frame,
        landmarks,
        p["right_hip"],
        p["right_shoulder"],
        p["right_elbow"],
        "R Shoulder",
    )

    draw_angle(
        frame,
        landmarks,
        p["right_shoulder"],
        p["right_elbow"],
        p["right_wrist"],
        "R Elbow",
    )

    draw_angle(
        frame,
        landmarks,
        p["right_shoulder"],
        p["right_hip"],
        p["right_knee"],
        "R Hip",
    )

    draw_angle(
        frame,
        landmarks,
        p["right_hip"],
        p["right_knee"],
        p["right_ankle"],
        "R Knee",
    )

    draw_angle(
        frame,
        landmarks,
        p["right_knee"],
        p["right_ankle"],
        p["right_foot_index"],
        "R Ankle",
    )


# =========================================================
# Hand描画
# =========================================================

def draw_hand_landmarks(
    frame,
    hand_landmarks,
):
    """
    21個の手指ランドマークと
    接続線を動画上に描画する。
    """

    height, width = (
        frame.shape[:2]
    )


    # -----------------------------------------------------
    # 接続線
    # -----------------------------------------------------

    for start_index, end_index in (
        HAND_CONNECTIONS
    ):

        start = landmark_to_pixel(
            hand_landmarks[
                start_index
            ],
            width,
            height,
        )

        end = landmark_to_pixel(
            hand_landmarks[
                end_index
            ],
            width,
            height,
        )

        cv2.line(
            frame,
            start,
            end,
            (
                255,
                180,
                50,
            ),
            2,
            cv2.LINE_AA,
        )


    # -----------------------------------------------------
    # 関節点
    # -----------------------------------------------------

    for landmark in (
        hand_landmarks
    ):

        point = landmark_to_pixel(
            landmark,
            width,
            height,
        )

        cv2.circle(
            frame,
            point,
            3,
            (
                255,
                255,
                0,
            ),
            -1,
            cv2.LINE_AA,
        )


# =========================================================
# 手関節軌跡描画
# =========================================================

def draw_trajectory(
    frame,
    trajectory,
    color,
):
    """
    手関節の過去の位置を線として表示する。
    """

    if len(
        trajectory
    ) < 2:

        return


    points = np.array(
        trajectory,
        dtype=np.int32,
    )


    cv2.polylines(
        frame,
        [
            points
        ],
        False,
        color,
        2,
        cv2.LINE_AA,
    )


# =========================================================
# 解析動画生成
# =========================================================

def create_annotated_video(
    video_path,
    output_path,
    phases=None,
    show_angles=True,
    show_hands=True,
    show_trajectory=True,
):
    """
    元動画へ、

    ・全身骨格
    ・関節点
    ・2D投影角
    ・手指ランドマーク
    ・手関節軌跡
    ・工程名
    ・経過時間

    を重ねたMP4動画を生成する。
    """

    if phases is None:
        phases = []


    video_path = Path(
        video_path
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------
    # モデル確認
    # -----------------------------------------------------

    if not POSE_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Poseモデルが見つかりません。"
        )


    if not HAND_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Handモデルが見つかりません。"
        )


    # -----------------------------------------------------
    # 動画情報
    # -----------------------------------------------------

    video_info = get_video_info(
        video_path
    )

    fps = float(
        video_info[
            "fps"
        ]
    )

    width = int(
        video_info[
            "width"
        ]
    )

    height = int(
        video_info[
            "height"
        ]
    )


    # -----------------------------------------------------
    # OpenCV読み込み
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        raise ValueError(
            "動画ファイルを開けませんでした。"
        )


    # -----------------------------------------------------
    # 一旦MJPEG AVIで作成
    # -----------------------------------------------------

    temporary_avi = (
        output_path.parent
        / (
            output_path.stem
            + "_temporary.avi"
        )
    )


    writer = cv2.VideoWriter(
        str(
            temporary_avi
        ),
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
        fps,
        (
            width,
            height,
        ),
    )


    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "解析動画のVideoWriterを"
            "初期化できませんでした。"
        )


    # -----------------------------------------------------
    # MediaPipe設定
    # -----------------------------------------------------

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    VisionRunningMode = (
        mp.tasks.vision.RunningMode
    )


    PoseLandmarker = (
        mp.tasks.vision.PoseLandmarker
    )

    PoseLandmarkerOptions = (
        mp.tasks.vision
        .PoseLandmarkerOptions
    )


    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision
        .HandLandmarkerOptions
    )


    pose_options = (
        PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(
                    POSE_MODEL_PATH
                )
            ),
            running_mode=(
                VisionRunningMode.VIDEO
            ),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


    hand_options = (
        HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(
                    HAND_MODEL_PATH
                )
            ),
            running_mode=(
                VisionRunningMode.VIDEO
            ),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


    # -----------------------------------------------------
    # 軌跡
    # -----------------------------------------------------

    left_wrist_trajectory = deque(
        maxlen=45
    )

    right_wrist_trajectory = deque(
        maxlen=45
    )


    frame_index = 0


    try:

        with (
            PoseLandmarker.create_from_options(
                pose_options
            ) as pose_landmarker,

            HandLandmarker.create_from_options(
                hand_options
            ) as hand_landmarker,
        ):

            while True:

                success, frame = (
                    cap.read()
                )

                if not success:
                    break


                time_seconds = (
                    frame_index
                    / fps
                )


                timestamp_ms = int(
                    round(
                        time_seconds
                        * 1000
                    )
                )


                rgb_frame = (
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )
                )


                mp_image = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=rgb_frame,
                )


                # =====================================
                # Pose解析
                # =====================================

                pose_result = (
                    pose_landmarker
                    .detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )
                )


                if (
                    pose_result
                    .pose_landmarks
                ):

                    landmarks = (
                        pose_result
                        .pose_landmarks[0]
                    )


                    draw_pose_skeleton(
                        frame,
                        landmarks,
                    )


                    if show_angles:

                        draw_pose_angles(
                            frame,
                            landmarks,
                        )


                    # ---------------------------------
                    # 手関節軌跡
                    # ---------------------------------

                    if show_trajectory:

                        left_wrist = (
                            landmarks[
                                POSE_INDEX[
                                    "left_wrist"
                                ]
                            ]
                        )

                        right_wrist = (
                            landmarks[
                                POSE_INDEX[
                                    "right_wrist"
                                ]
                            ]
                        )


                        if is_landmark_visible(
                            left_wrist
                        ):

                            left_point = (
                                landmark_to_pixel(
                                    left_wrist,
                                    width,
                                    height,
                                )
                            )

                            left_wrist_trajectory.append(
                                left_point
                            )


                        if is_landmark_visible(
                            right_wrist
                        ):

                            right_point = (
                                landmark_to_pixel(
                                    right_wrist,
                                    width,
                                    height,
                                )
                            )

                            right_wrist_trajectory.append(
                                right_point
                            )


                        draw_trajectory(
                            frame,
                            left_wrist_trajectory,
                            (
                                80,
                                220,
                                80,
                            ),
                        )


                        draw_trajectory(
                            frame,
                            right_wrist_trajectory,
                            (
                                80,
                                170,
                                255,
                            ),
                        )


                # =====================================
                # Hand解析
                # =====================================

                if show_hands:

                    hand_result = (
                        hand_landmarker
                        .detect_for_video(
                            mp_image,
                            timestamp_ms,
                        )
                    )


                    if (
                        hand_result
                        .hand_landmarks
                    ):

                        for hand_landmarks in (
                            hand_result
                            .hand_landmarks
                        ):

                            draw_hand_landmarks(
                                frame,
                                hand_landmarks,
                            )


                # =====================================
                # 工程
                # =====================================

                phase_name = get_phase(
                    time_seconds,
                    phases,
                )


                # -------------------------------------
                # 上部背景
                # -------------------------------------

                cv2.rectangle(
                    frame,
                    (
                        0,
                        0,
                    ),
                    (
                        width,
                        65,
                    ),
                    (
                        0,
                        0,
                        0,
                    ),
                    -1,
                )


                # -------------------------------------
                # 工程名
                # -------------------------------------

                cv2.putText(
                    frame,
                    f"Phase: {phase_name}",
                    (
                        20,
                        27,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )


                # -------------------------------------
                # 時刻
                # -------------------------------------

                cv2.putText(
                    frame,
                    f"Time: {time_seconds:.2f} s",
                    (
                        20,
                        55,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )


                # -------------------------------------
                # 注意表記
                # -------------------------------------

                cv2.putText(
                    frame,
                    "2D projected angles",
                    (
                        width - 240,
                        55,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (
                        220,
                        220,
                        220,
                    ),
                    1,
                    cv2.LINE_AA,
                )


                writer.write(
                    frame
                )

                frame_index += 1


    finally:

        cap.release()
        writer.release()


    # =====================================================
    # AVI → H.264 MP4
    # =====================================================

    ffmpeg_executable = (
        imageio_ffmpeg
        .get_ffmpeg_exe()
    )


    command = [
        ffmpeg_executable,
        "-y",

        "-i",
        str(
            temporary_avi
        ),

        "-c:v",
        "libx264",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        "-an",

        str(
            output_path
        ),
    ]


    try:

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    except subprocess.CalledProcessError as e:

        error_text = (
            e.stderr
            .decode(
                "utf-8",
                errors="ignore",
            )
        )

        raise RuntimeError(
            "MP4への変換に失敗しました。\n"
            + error_text[-1500:]
        )


    finally:

        if temporary_avi.exists():

            temporary_avi.unlink()


    return output_path