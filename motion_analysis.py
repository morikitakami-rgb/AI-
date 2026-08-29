from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# =========================================================
# OT Practice Copilot
# 動作解析基盤
# =========================================================


POSE_MODEL_PATH = Path(
    "models/pose_landmarker_full.task"
)

HAND_MODEL_PATH = Path(
    "models/hand_landmarker.task"
)


# =========================================================
# MediaPipe Poseのランドマーク番号
# =========================================================

POSE_INDEX = {
    "nose": 0,

    "left_shoulder": 11,
    "right_shoulder": 12,

    "left_elbow": 13,
    "right_elbow": 14,

    "left_wrist": 15,
    "right_wrist": 16,

    "left_hip": 23,
    "right_hip": 24,

    "left_knee": 25,
    "right_knee": 26,

    "left_ankle": 27,
    "right_ankle": 28,

    "left_heel": 29,
    "right_heel": 30,

    "left_foot_index": 31,
    "right_foot_index": 32,
}


# =========================================================
# 角度計算
# =========================================================

def calculate_angle_2d(
    point_a,
    point_b,
    point_c,
):
    """
    A-B-C の角度を計算する。
    Bが角度の中心。

    単眼動画上の2D投影角であり、
    臨床ROM値ではない。
    """

    a = np.array(
        [point_a.x, point_a.y],
        dtype=float,
    )

    b = np.array(
        [point_b.x, point_b.y],
        dtype=float,
    )

    c = np.array(
        [point_c.x, point_c.y],
        dtype=float,
    )

    vector_ba = a - b
    vector_bc = c - b

    denominator = (
        np.linalg.norm(vector_ba)
        * np.linalg.norm(vector_bc)
    )

    if denominator == 0:
        return np.nan

    cosine = (
        np.dot(
            vector_ba,
            vector_bc,
        )
        / denominator
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    angle = np.degrees(
        np.arccos(cosine)
    )

    return float(angle)


# =========================================================
# 体幹傾斜
# =========================================================

def calculate_trunk_tilt(
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip,
):
    """
    肩中点と股関節中点を結んだ線の
    鉛直方向に対する傾斜角を求める。
    """

    shoulder_mid = np.array(
        [
            (
                left_shoulder.x
                + right_shoulder.x
            )
            / 2,
            (
                left_shoulder.y
                + right_shoulder.y
            )
            / 2,
        ]
    )

    hip_mid = np.array(
        [
            (
                left_hip.x
                + right_hip.x
            )
            / 2,
            (
                left_hip.y
                + right_hip.y
            )
            / 2,
        ]
    )

    dx = (
        shoulder_mid[0]
        - hip_mid[0]
    )

    dy = (
        shoulder_mid[1]
        - hip_mid[1]
    )

    if dx == 0 and dy == 0:
        return np.nan

    angle = np.degrees(
        np.arctan2(
            abs(dx),
            abs(dy),
        )
    )

    return float(angle)


# =========================================================
# 手指
# =========================================================

def calculate_pinch_distance(
    hand_landmarks,
):
    """
    母指先端と示指先端の距離を計算。

    手の大きさの影響を減らすため、
    手関節－中指MCP距離で正規化する。
    """

    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_mcp = hand_landmarks[9]

    wrist_xy = np.array(
        [wrist.x, wrist.y]
    )

    thumb_xy = np.array(
        [
            thumb_tip.x,
            thumb_tip.y,
        ]
    )

    index_xy = np.array(
        [
            index_tip.x,
            index_tip.y,
        ]
    )

    middle_xy = np.array(
        [
            middle_mcp.x,
            middle_mcp.y,
        ]
    )

    pinch_distance = np.linalg.norm(
        thumb_xy - index_xy
    )

    hand_size = np.linalg.norm(
        wrist_xy - middle_xy
    )

    if hand_size == 0:
        return np.nan

    return float(
        pinch_distance
        / hand_size
    )


# =========================================================
# 動画情報
# =========================================================

def get_video_info(
    video_path,
):
    """
    FPS、フレーム数、動画時間等を取得。
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise ValueError(
            "動画ファイルを開けませんでした。"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    if fps <= 0:

        raise ValueError(
            "動画のFPSを取得できませんでした。"
        )

    duration = (
        frame_count / fps
        if frame_count > 0
        else 0
    )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }


# =========================================================
# 工程判定
# =========================================================

def get_phase(
    time_seconds,
    phases,
):
    """
    現在時刻がどの工程に該当するか返す。

    phasesの例：
    [
        {
            "name": "リーチ",
            "start": 1.0,
            "end": 2.5
        }
    ]
    """

    for phase in phases:

        if (
            phase["start"]
            <= time_seconds
            < phase["end"]
        ):

            return phase["name"]

    return "工程外"


# =========================================================
# 1フレームのPose数値化
# =========================================================

def extract_pose_metrics(
    landmarks,
):
    """
    Poseランドマークから
    全身の基本指標を計算する。
    """

    p = POSE_INDEX

    ls = landmarks[
        p["left_shoulder"]
    ]

    rs = landmarks[
        p["right_shoulder"]
    ]

    le = landmarks[
        p["left_elbow"]
    ]

    re = landmarks[
        p["right_elbow"]
    ]

    lw = landmarks[
        p["left_wrist"]
    ]

    rw = landmarks[
        p["right_wrist"]
    ]

    lh = landmarks[
        p["left_hip"]
    ]

    rh = landmarks[
        p["right_hip"]
    ]

    lk = landmarks[
        p["left_knee"]
    ]

    rk = landmarks[
        p["right_knee"]
    ]

    la = landmarks[
        p["left_ankle"]
    ]

    ra = landmarks[
        p["right_ankle"]
    ]

    lf = landmarks[
        p["left_foot_index"]
    ]

    rf = landmarks[
        p["right_foot_index"]
    ]


    return {

        # -----------------------------------------
        # 上肢
        # -----------------------------------------

        "left_shoulder_proj_deg":
            calculate_angle_2d(
                lh,
                ls,
                le,
            ),

        "right_shoulder_proj_deg":
            calculate_angle_2d(
                rh,
                rs,
                re,
            ),

        "left_elbow_proj_deg":
            calculate_angle_2d(
                ls,
                le,
                lw,
            ),

        "right_elbow_proj_deg":
            calculate_angle_2d(
                rs,
                re,
                rw,
            ),


        # -----------------------------------------
        # 下肢
        # -----------------------------------------

        "left_hip_proj_deg":
            calculate_angle_2d(
                ls,
                lh,
                lk,
            ),

        "right_hip_proj_deg":
            calculate_angle_2d(
                rs,
                rh,
                rk,
            ),

        "left_knee_proj_deg":
            calculate_angle_2d(
                lh,
                lk,
                la,
            ),

        "right_knee_proj_deg":
            calculate_angle_2d(
                rh,
                rk,
                ra,
            ),

        "left_ankle_proj_deg":
            calculate_angle_2d(
                lk,
                la,
                lf,
            ),

        "right_ankle_proj_deg":
            calculate_angle_2d(
                rk,
                ra,
                rf,
            ),


        # -----------------------------------------
        # 体幹
        # -----------------------------------------

        "trunk_tilt_deg":
            calculate_trunk_tilt(
                ls,
                rs,
                lh,
                rh,
            ),


        # -----------------------------------------
        # 軌跡用座標
        # -----------------------------------------

        "left_wrist_x": lw.x,
        "left_wrist_y": lw.y,

        "right_wrist_x": rw.x,
        "right_wrist_y": rw.y,

        "left_ankle_x": la.x,
        "left_ankle_y": la.y,

        "right_ankle_x": ra.x,
        "right_ankle_y": ra.y,
    }


# =========================================================
# 動画解析
# =========================================================

def analyze_video(
    video_path,
    phases=None,
    target_fps=30,
):
    """
    動画をPose + Hand Landmarkerで解析し、
    フレームごとのDataFrameを返す。
    """

    if phases is None:
        phases = []


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


    video_info = get_video_info(
        video_path
    )

    source_fps = video_info[
        "fps"
    ]

    target_fps = min(
        float(target_fps),
        source_fps,
    )

    frame_step = max(
        1,
        int(
            round(
                source_fps
                / target_fps
            )
        ),
    )


    # =====================================================
    # MediaPipe設定
    # =====================================================

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


    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise ValueError(
            "動画を開けませんでした。"
        )


    rows = []


    with (
        PoseLandmarker.create_from_options(
            pose_options
        ) as pose_landmarker,

        HandLandmarker.create_from_options(
            hand_options
        ) as hand_landmarker,
    ):

        frame_index = 0

        while True:

            success, frame = cap.read()

            if not success:
                break


            # -----------------------------------------
            # 間引き
            # -----------------------------------------

            if (
                frame_index
                % frame_step
                != 0
            ):

                frame_index += 1
                continue


            time_seconds = (
                frame_index
                / source_fps
            )

            timestamp_ms = int(
                round(
                    time_seconds
                    * 1000
                )
            )


            # OpenCV BGR → RGB
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


            # -----------------------------------------
            # Pose
            # -----------------------------------------

            pose_result = (
                pose_landmarker
                .detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )


            # -----------------------------------------
            # Hand
            # -----------------------------------------

            hand_result = (
                hand_landmarker
                .detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )


            row = {
                "frame": frame_index,

                "time_s":
                    time_seconds,

                "phase":
                    get_phase(
                        time_seconds,
                        phases,
                    ),
            }


            # -----------------------------------------
            # Pose数値
            # -----------------------------------------

            if pose_result.pose_landmarks:

                metrics = (
                    extract_pose_metrics(
                        pose_result
                        .pose_landmarks[0]
                    )
                )

                row.update(
                    metrics
                )


            # -----------------------------------------
            # Hand
            # -----------------------------------------

            row[
                "left_pinch_norm"
            ] = np.nan

            row[
                "right_pinch_norm"
            ] = np.nan


            if hand_result.hand_landmarks:

                for (
                    hand_landmarks,
                    handedness,
                ) in zip(
                    hand_result.hand_landmarks,
                    hand_result.handedness,
                ):

                    if not handedness:
                        continue

                    hand_side = (
                        handedness[0]
                        .category_name
                        .lower()
                    )

                    pinch = (
                        calculate_pinch_distance(
                            hand_landmarks
                        )
                    )

                    if hand_side == "left":

                        row[
                            "left_pinch_norm"
                        ] = pinch

                    elif hand_side == "right":

                        row[
                            "right_pinch_norm"
                        ] = pinch


            rows.append(
                row
            )

            frame_index += 1


    cap.release()


    df = pd.DataFrame(
        rows
    )


    if df.empty:

        raise ValueError(
            "解析可能なフレームがありませんでした。"
        )


    # =====================================================
    # 軌跡・速度計算
    # =====================================================

    delta_time = df[
        "time_s"
    ].diff()


    for side in [
        "left",
        "right",
    ]:

        x_column = (
            f"{side}_wrist_x"
        )

        y_column = (
            f"{side}_wrist_y"
        )


        if (
            x_column in df.columns
            and y_column in df.columns
        ):

            dx = df[
                x_column
            ].diff()

            dy = df[
                y_column
            ].diff()

            distance = np.sqrt(
                dx ** 2
                + dy ** 2
            )

            df[
                f"{side}_wrist_distance_norm"
            ] = distance

            df[
                f"{side}_wrist_speed_norm_s"
            ] = (
                distance
                / delta_time
            )


    return df, video_info


# =========================================================
# 工程別集計
# =========================================================

def summarize_by_phase(
    df,
):
    """
    工程ごとに基本指標をまとめる。
    """

    if (
        "phase"
        not in df.columns
    ):

        return pd.DataFrame()


    summary_rows = []


    for phase_name, group in (
        df.groupby(
            "phase"
        )
    ):

        if phase_name == "工程外":
            continue


        row = {
            "工程": phase_name,

            "開始秒":
                group["time_s"].min(),

            "終了秒":
                group["time_s"].max(),

            "解析フレーム数":
                len(group),
        }


        columns_to_average = [
            "left_shoulder_proj_deg",
            "right_shoulder_proj_deg",

            "left_elbow_proj_deg",
            "right_elbow_proj_deg",

            "left_hip_proj_deg",
            "right_hip_proj_deg",

            "left_knee_proj_deg",
            "right_knee_proj_deg",

            "left_ankle_proj_deg",
            "right_ankle_proj_deg",

            "trunk_tilt_deg",

            "left_wrist_speed_norm_s",
            "right_wrist_speed_norm_s",

            "left_pinch_norm",
            "right_pinch_norm",
        ]


        for column in (
            columns_to_average
        ):

            if column in group.columns:

                row[
                    f"{column}_mean"
                ] = (
                    group[column]
                    .mean()
                )

                row[
                    f"{column}_max"
                ] = (
                    group[column]
                    .max()
                )


        summary_rows.append(
            row
        )


    return pd.DataFrame(
        summary_rows
    )