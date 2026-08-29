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
    POSE_TRAJECTORY_POINTS,
    HAND_POINT_INDICES,
    TRAJECTORY_POINT_LABELS,
    get_trajectory_points,
    calculate_angle_2d,
    get_phase,
    get_video_info,
)


# =========================================================
# OT Practice Copilot
# 全身軌跡・骨格オーバーレイ動画
# =========================================================


# =========================================================
# Pose骨格接続
# =========================================================

POSE_CONNECTIONS = [
    # 肩
    (11, 12),

    # 左上肢
    (11, 13),
    (13, 15),

    # 右上肢
    (12, 14),
    (14, 16),

    # 体幹
    (11, 23),
    (12, 24),
    (23, 24),

    # 左下肢
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),

    # 右下肢
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
]


# =========================================================
# Hand骨格接続
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
# 軌跡の色
# =========================================================

TRAJECTORY_COLORS = [
    (0, 255, 255),
    (255, 150, 0),
    (0, 255, 100),
    (255, 0, 255),
    (100, 200, 255),
    (255, 100, 100),
    (100, 255, 255),
    (200, 100, 255),
    (100, 255, 150),
    (255, 200, 100),
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
    MediaPipe正規化座標を
    ピクセル座標へ変換する。
    """

    x = int(
        np.clip(
            landmark.x,
            0.0,
            1.0,
        )
        * width
    )

    y = int(
        np.clip(
            landmark.y,
            0.0,
            1.0,
        )
        * height
    )

    return (
        x,
        y,
    )


# =========================================================
# Poseランドマーク可視性
# =========================================================

def is_landmark_visible(
    landmark,
    threshold=0.4,
):
    """
    visibilityが一定以上か確認する。
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
# Poseの軌跡点座標
# =========================================================

def get_pose_point_pixel(
    point_name,
    landmarks,
    width,
    height,
    visibility_threshold=0.4,
):
    """
    Pose上の軌跡対象点の
    ピクセル座標を取得する。

    肩中央・骨盤中央も計算する。
    """

    # -----------------------------------------------------
    # 肩中央
    # -----------------------------------------------------

    if point_name == "shoulder_mid":

        left = landmarks[
            POSE_INDEX[
                "left_shoulder"
            ]
        ]

        right = landmarks[
            POSE_INDEX[
                "right_shoulder"
            ]
        ]

        if not (
            is_landmark_visible(
                left,
                visibility_threshold,
            )
            and
            is_landmark_visible(
                right,
                visibility_threshold,
            )
        ):
            return None

        x = (
            left.x
            + right.x
        ) / 2

        y = (
            left.y
            + right.y
        ) / 2

        return (
            int(
                np.clip(
                    x,
                    0.0,
                    1.0,
                )
                * width
            ),
            int(
                np.clip(
                    y,
                    0.0,
                    1.0,
                )
                * height
            ),
        )


    # -----------------------------------------------------
    # 骨盤中央
    # -----------------------------------------------------

    if point_name == "pelvis_mid":

        left = landmarks[
            POSE_INDEX[
                "left_hip"
            ]
        ]

        right = landmarks[
            POSE_INDEX[
                "right_hip"
            ]
        ]

        if not (
            is_landmark_visible(
                left,
                visibility_threshold,
            )
            and
            is_landmark_visible(
                right,
                visibility_threshold,
            )
        ):
            return None

        x = (
            left.x
            + right.x
        ) / 2

        y = (
            left.y
            + right.y
        ) / 2

        return (
            int(
                np.clip(
                    x,
                    0.0,
                    1.0,
                )
                * width
            ),
            int(
                np.clip(
                    y,
                    0.0,
                    1.0,
                )
                * height
            ),
        )


    # -----------------------------------------------------
    # 通常Poseランドマーク
    # -----------------------------------------------------

    if point_name not in (
        POSE_TRAJECTORY_POINTS
    ):

        return None


    index = (
        POSE_TRAJECTORY_POINTS[
            point_name
        ]
    )


    landmark = landmarks[
        index
    ]


    if not is_landmark_visible(
        landmark,
        visibility_threshold,
    ):

        return None


    return landmark_to_pixel(
        landmark,
        width,
        height,
    )


# =========================================================
# Handの軌跡点座標
# =========================================================

def get_hand_point_pixel(
    full_point_name,
    side,
    hand_landmarks,
    width,
    height,
):
    """
    left_index_tip などの名前から
    Hand Landmarker上の点を取得する。
    """

    prefix = (
        f"{side}_"
    )

    if not full_point_name.startswith(
        prefix
    ):

        return None


    local_name = (
        full_point_name[
            len(prefix):
        ]
    )


    if local_name not in (
        HAND_POINT_INDICES
    ):

        return None


    index = (
        HAND_POINT_INDICES[
            local_name
        ]
    )


    return landmark_to_pixel(
        hand_landmarks[
            index
        ],
        width,
        height,
    )


# =========================================================
# Pose骨格描画
# =========================================================

def draw_pose_skeleton(
    frame,
    landmarks,
):
    """
    全身骨格を描画する。
    """

    height, width = (
        frame.shape[:2]
    )


    for (
        start_index,
        end_index,
    ) in POSE_CONNECTIONS:

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


        start = landmark_to_pixel(
            start_landmark,
            width,
            height,
        )

        end = landmark_to_pixel(
            end_landmark,
            width,
            height,
        )


        cv2.line(
            frame,
            start,
            end,
            (
                220,
                220,
                220,
            ),
            2,
            cv2.LINE_AA,
        )


    # -----------------------------------------------------
    # 関節点
    # -----------------------------------------------------

    for index in set(
        point
        for connection in POSE_CONNECTIONS
        for point in connection
    ):

        landmark = (
            landmarks[
                index
            ]
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


        cv2.circle(
            frame,
            point,
            5,
            (
                0,
                255,
                180,
            ),
            -1,
            cv2.LINE_AA,
        )


# =========================================================
# Hand骨格描画
# =========================================================

def draw_hand_skeleton(
    frame,
    hand_landmarks,
):
    """
    手指21点と接続線を描画する。
    """

    height, width = (
        frame.shape[:2]
    )


    for (
        start_index,
        end_index,
    ) in HAND_CONNECTIONS:

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
                40,
            ),
            2,
            cv2.LINE_AA,
        )


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
                0,
                255,
                255,
            ),
            -1,
            cv2.LINE_AA,
        )


# =========================================================
# 角度表示
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
    指定した3点から2D投影角を計算し、
    関節付近に表示する。
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
        f"{label}: "
        f"{angle:.0f} deg"
    )


    # 黒い縁取り
    cv2.putText(
        frame,
        text,
        (
            x + 7,
            y - 7,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (
            0,
            0,
            0,
        ),
        3,
        cv2.LINE_AA,
    )


    cv2.putText(
        frame,
        text,
        (
            x + 7,
            y - 7,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )


# =========================================================
# 全身主要関節角度
# =========================================================

def draw_pose_angles(
    frame,
    landmarks,
):
    """
    肩・肘・股・膝・足関節の
    2D投影角を表示する。
    """

    p = POSE_INDEX


    # 左上肢
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


    # 右上肢
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


    # 左下肢
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


    # 右下肢
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
# 軌跡描画
# =========================================================

def draw_trajectory(
    frame,
    points,
    color,
    label=None,
):
    """
    1つの部位の軌跡を描画する。
    """

    if len(
        points
    ) < 2:

        return


    array = np.array(
        points,
        dtype=np.int32,
    )


    cv2.polylines(
        frame,
        [
            array
        ],
        False,
        color,
        2,
        cv2.LINE_AA,
    )


    # 現在位置
    last_point = points[
        -1
    ]


    cv2.circle(
        frame,
        last_point,
        6,
        color,
        -1,
        cv2.LINE_AA,
    )


    if label:

        cv2.putText(
            frame,
            label,
            (
                last_point[0] + 5,
                last_point[1] + 15,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


# =========================================================
# オーバーレイ動画作成
# =========================================================

def create_annotated_video(
    video_path,
    output_path,
    phases=None,
    trajectory_preset="全身",
    trajectory_points=None,
    trajectory_duration_s=1.5,
    show_skeleton=True,
    show_angles=True,
    show_hands=True,
    show_trajectory=True,
    show_trajectory_labels=False,
    reset_trajectory_each_phase=False,
):
    """
    元動画へ、

    ・全身骨格
    ・関節点
    ・2D投影角
    ・手指ランドマーク
    ・選択した全身軌跡
    ・工程
    ・時刻

    を重ねたMP4を生成する。


    trajectory_preset:
        上肢
        体幹
        下肢
        手指
        リーチ
        立ち上がり
        歩行
        全身
        全身＋手指


    trajectory_duration_s:
        例 1.5 → 直近1.5秒
        None → 動画全体
    """

    if phases is None:
        phases = []


    # =====================================================
    # 軌跡対象
    # =====================================================

    if trajectory_points is None:

        trajectory_points = (
            get_trajectory_points(
                trajectory_preset
            )
        )


    if not trajectory_points:

        trajectory_points = (
            get_trajectory_points(
                "全身"
            )
        )


    # =====================================================
    # ファイル
    # =====================================================

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


    if not POSE_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Poseモデルが見つかりません。"
        )


    if not HAND_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Handモデルが見つかりません。"
        )


    # =====================================================
    # 動画情報
    # =====================================================

    video_info = (
        get_video_info(
            video_path
        )
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


    # =====================================================
    # 軌跡保持フレーム数
    # =====================================================

    if trajectory_duration_s is None:

        trajectory_maxlen = None

    else:

        trajectory_maxlen = max(
            2,
            int(
                round(
                    fps
                    * float(
                        trajectory_duration_s
                    )
                )
            ),
        )


    trajectories = {}


    for point_name in (
        trajectory_points
    ):

        trajectories[
            point_name
        ] = deque(
            maxlen=(
                trajectory_maxlen
            )
        )


    # =====================================================
    # OpenCV
    # =====================================================

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )


    if not cap.isOpened():

        raise ValueError(
            "動画ファイルを開けませんでした。"
        )


    # =====================================================
    # 一時AVI
    # =====================================================

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
            "VideoWriterを"
            "初期化できませんでした。"
        )


    # =====================================================
    # MediaPipe
    # =====================================================

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    VisionRunningMode = (
        mp.tasks.vision.RunningMode
    )


    PoseLandmarker = (
        mp.tasks.vision
        .PoseLandmarker
    )

    PoseLandmarkerOptions = (
        mp.tasks.vision
        .PoseLandmarkerOptions
    )


    HandLandmarker = (
        mp.tasks.vision
        .HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision
        .HandLandmarkerOptions
    )


    pose_options = (
        PoseLandmarkerOptions(

            base_options=(
                BaseOptions(
                    model_asset_path=str(
                        POSE_MODEL_PATH
                    )
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

            base_options=(
                BaseOptions(
                    model_asset_path=str(
                        HAND_MODEL_PATH
                    )
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


    frame_index = 0
    previous_phase = None


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


                current_phase = (
                    get_phase(
                        time_seconds,
                        phases,
                    )
                )


                # =====================================
                # 工程変化で軌跡リセット
                # =====================================

                if (
                    reset_trajectory_each_phase
                    and
                    previous_phase is not None
                    and
                    current_phase
                    != previous_phase
                ):

                    for trajectory in (
                        trajectories.values()
                    ):

                        trajectory.clear()


                previous_phase = (
                    current_phase
                )


                # =====================================
                # MediaPipe画像
                # =====================================

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
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


                # =====================================
                # Hand解析
                # =====================================

                hand_result = (
                    hand_landmarker
                    .detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )
                )


                pose_landmarks = None


                if (
                    pose_result
                    .pose_landmarks
                ):

                    pose_landmarks = (
                        pose_result
                        .pose_landmarks[0]
                    )


                    # ---------------------------------
                    # 骨格
                    # ---------------------------------

                    if show_skeleton:

                        draw_pose_skeleton(
                            frame,
                            pose_landmarks,
                        )


                    # ---------------------------------
                    # 角度
                    # ---------------------------------

                    if show_angles:

                        draw_pose_angles(
                            frame,
                            pose_landmarks,
                        )


                    # ---------------------------------
                    # Pose軌跡更新
                    # ---------------------------------

                    if show_trajectory:

                        for point_name in (
                            trajectory_points
                        ):

                            point = (
                                get_pose_point_pixel(
                                    point_name,
                                    pose_landmarks,
                                    width,
                                    height,
                                )
                            )


                            if point is not None:

                                trajectories[
                                    point_name
                                ].append(
                                    point
                                )


                # =====================================
                # Hand
                # =====================================

                if (
                    hand_result
                    .hand_landmarks
                ):


                    for (
                        hand_landmarks,
                        handedness,
                    ) in zip(
                        hand_result
                        .hand_landmarks,
                        hand_result
                        .handedness,
                    ):


                        if not handedness:
                            continue


                        side = (
                            handedness[0]
                            .category_name
                            .lower()
                        )


                        if side not in [
                            "left",
                            "right",
                        ]:
                            continue


                        if show_hands:

                            draw_hand_skeleton(
                                frame,
                                hand_landmarks,
                            )


                        # -----------------------------
                        # 手指軌跡
                        # -----------------------------

                        if show_trajectory:

                            for point_name in (
                                trajectory_points
                            ):

                                point = (
                                    get_hand_point_pixel(
                                        point_name,
                                        side,
                                        hand_landmarks,
                                        width,
                                        height,
                                    )
                                )


                                if point is not None:

                                    trajectories[
                                        point_name
                                    ].append(
                                        point
                                    )


                # =====================================
                # 軌跡描画
                # =====================================

                if show_trajectory:

                    for index, point_name in enumerate(
                        trajectory_points
                    ):

                        color = (
                            TRAJECTORY_COLORS[
                                index
                                % len(
                                    TRAJECTORY_COLORS
                                )
                            ]
                        )


                        if show_trajectory_labels:

                            label = (
                                TRAJECTORY_POINT_LABELS
                                .get(
                                    point_name,
                                    point_name,
                                )
                            )

                        else:

                            label = None


                        draw_trajectory(
                            frame,
                            trajectories[
                                point_name
                            ],
                            color,
                            label=label,
                        )


                # =====================================
                # 上部情報
                # =====================================

                cv2.rectangle(
                    frame,
                    (
                        0,
                        0,
                    ),
                    (
                        width,
                        70,
                    ),
                    (
                        0,
                        0,
                        0,
                    ),
                    -1,
                )


                # OpenCV標準フォントでは
                # 日本語表示が安定しないため、
                # Phase名はASCII推奨。
                cv2.putText(
                    frame,
                    (
                        f"Time: "
                        f"{time_seconds:.2f} s"
                    ),
                    (
                        20,
                        28,
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


                cv2.putText(
                    frame,
                    (
                        f"Preset: "
                        f"{trajectory_preset}"
                        if trajectory_preset.isascii()
                        else "Trajectory analysis"
                    ),
                    (
                        20,
                        57,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (
                        230,
                        230,
                        230,
                    ),
                    1,
                    cv2.LINE_AA,
                )


                cv2.putText(
                    frame,
                    "2D projected angles",
                    (
                        max(
                            20,
                            width - 230,
                        ),
                        57,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
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

        "-preset",
        "fast",

        "-crf",
        "22",

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
            + error_text[
                -2000:
            ]
        )


    finally:

        if temporary_avi.exists():

            temporary_avi.unlink()


    return output_path