from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# =========================================================
# OT Practice Copilot
# 全身・手指 動作解析基盤
# =========================================================


# =========================================================
# 基本パス
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

POSE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "pose_landmarker_full.task"
)

HAND_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)


# =========================================================
# MediaPipe Pose ランドマーク番号
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
# 軌跡解析対象
# =========================================================

POSE_TRAJECTORY_POINTS = {
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
# 手指軌跡
# =========================================================

HAND_POINT_INDICES = {
    "hand_wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}


# =========================================================
# 表示名
# =========================================================

TRAJECTORY_POINT_LABELS = {
    # 上肢
    "left_shoulder": "左肩",
    "right_shoulder": "右肩",
    "left_elbow": "左肘",
    "right_elbow": "右肘",
    "left_wrist": "左手関節",
    "right_wrist": "右手関節",

    # 体幹
    "shoulder_mid": "肩中央",
    "pelvis_mid": "骨盤中央",

    # 下肢
    "left_hip": "左股関節",
    "right_hip": "右股関節",
    "left_knee": "左膝",
    "right_knee": "右膝",
    "left_ankle": "左足関節",
    "right_ankle": "右足関節",
    "left_heel": "左踵",
    "right_heel": "右踵",
    "left_foot_index": "左足趾",
    "right_foot_index": "右足趾",

    # 左手
    "left_hand_wrist": "左手部手関節",
    "left_thumb_tip": "左母指先端",
    "left_index_tip": "左示指先端",
    "left_middle_tip": "左中指先端",
    "left_ring_tip": "左環指先端",
    "left_pinky_tip": "左小指先端",

    # 右手
    "right_hand_wrist": "右手部手関節",
    "right_thumb_tip": "右母指先端",
    "right_index_tip": "右示指先端",
    "right_middle_tip": "右中指先端",
    "right_ring_tip": "右環指先端",
    "right_pinky_tip": "右小指先端",
}


# =========================================================
# 軌跡プリセット
# =========================================================

TRAJECTORY_PRESETS = {

    "上肢": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "shoulder_mid",
    ],

    "体幹": [
        "left_shoulder",
        "right_shoulder",
        "shoulder_mid",
        "left_hip",
        "right_hip",
        "pelvis_mid",
    ],

    "下肢": [
        "left_hip",
        "right_hip",
        "pelvis_mid",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ],

    "手指": [
        "left_hand_wrist",
        "left_thumb_tip",
        "left_index_tip",
        "left_middle_tip",
        "left_ring_tip",
        "left_pinky_tip",

        "right_hand_wrist",
        "right_thumb_tip",
        "right_index_tip",
        "right_middle_tip",
        "right_ring_tip",
        "right_pinky_tip",
    ],

    "リーチ": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "shoulder_mid",
    ],

    "立ち上がり": [
        "shoulder_mid",
        "pelvis_mid",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],

    "歩行": [
        "pelvis_mid",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ],

    "全身": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",

        "shoulder_mid",
        "pelvis_mid",

        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ],

    "全身＋手指": list(
        TRAJECTORY_POINT_LABELS.keys()
    ),
}


# =========================================================
# プリセット取得
# =========================================================

def get_trajectory_points(
    preset_name,
):
    """
    プリセット名から軌跡対象点を返す。
    """

    return TRAJECTORY_PRESETS.get(
        preset_name,
        []
    )


# =========================================================
# 2D角度
# =========================================================

def calculate_angle_2d(
    point_a,
    point_b,
    point_c,
):
    """
    A-B-C の2D投影角を計算する。

    Bが角度の中心。

    注意：
    単眼動画上の2D投影角であり、
    臨床ROMそのものではない。
    """

    a = np.array(
        [
            point_a.x,
            point_a.y,
        ],
        dtype=float,
    )

    b = np.array(
        [
            point_b.x,
            point_b.y,
        ],
        dtype=float,
    )

    c = np.array(
        [
            point_c.x,
            point_c.y,
        ],
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
        np.arccos(
            cosine
        )
    )

    return float(
        angle
    )


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
    肩中央と骨盤中央を結んだ線の
    鉛直方向に対する2D傾斜角を計算する。
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
        ],
        dtype=float,
    )

    pelvis_mid = np.array(
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
        ],
        dtype=float,
    )

    dx = (
        shoulder_mid[0]
        - pelvis_mid[0]
    )

    dy = (
        shoulder_mid[1]
        - pelvis_mid[1]
    )

    if (
        dx == 0
        and dy == 0
    ):
        return np.nan

    angle = np.degrees(
        np.arctan2(
            abs(dx),
            abs(dy),
        )
    )

    return float(
        angle
    )


# =========================================================
# 手指ピンチ距離
# =========================================================

def calculate_pinch_distance(
    hand_landmarks,
):
    """
    母指先端と示指先端の距離を計算する。

    手の大きさによる影響を軽減するため、
    手関節－中指MCP距離で正規化する。
    """

    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_mcp = hand_landmarks[9]

    wrist_xy = np.array(
        [
            wrist.x,
            wrist.y,
        ]
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
        thumb_xy
        - index_xy
    )

    hand_size = np.linalg.norm(
        wrist_xy
        - middle_xy
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
    FPS、フレーム数、動画時間、
    画面サイズを取得する。
    """

    cap = cv2.VideoCapture(
        str(
            video_path
        )
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
# 工程
# =========================================================

def get_phase(
    time_seconds,
    phases,
):
    """
    指定時刻がどの工程か判定する。
    """

    for phase in phases:

        if (
            phase["start"]
            <= time_seconds
            < phase["end"]
        ):

            return phase[
                "name"
            ]

    return "工程外"


# =========================================================
# Pose数値抽出
# =========================================================

def extract_pose_metrics(
    landmarks,
):
    """
    Poseランドマークから、

    ・主要関節2D投影角
    ・体幹傾斜
    ・主要全身座標
    ・肩中央
    ・骨盤中央

    を取得する。
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


    metrics = {

        # =========================================
        # 上肢2D投影角
        # =========================================

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


        # =========================================
        # 下肢2D投影角
        # =========================================

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


        # =========================================
        # 体幹
        # =========================================

        "trunk_tilt_deg":
            calculate_trunk_tilt(
                ls,
                rs,
                lh,
                rh,
            ),
    }


    # =====================================================
    # 全身主要点
    # =====================================================

    for (
        point_name,
        point_index,
    ) in POSE_TRAJECTORY_POINTS.items():

        landmark = landmarks[
            point_index
        ]

        metrics[
            f"{point_name}_x"
        ] = float(
            landmark.x
        )

        metrics[
            f"{point_name}_y"
        ] = float(
            landmark.y
        )

        metrics[
            f"{point_name}_z"
        ] = float(
            landmark.z
        )

        visibility = getattr(
            landmark,
            "visibility",
            np.nan,
        )

        metrics[
            f"{point_name}_visibility"
        ] = float(
            visibility
        )


    # =====================================================
    # 肩中央
    # =====================================================

    metrics[
        "shoulder_mid_x"
    ] = (
        ls.x
        + rs.x
    ) / 2

    metrics[
        "shoulder_mid_y"
    ] = (
        ls.y
        + rs.y
    ) / 2

    metrics[
        "shoulder_mid_z"
    ] = (
        ls.z
        + rs.z
    ) / 2


    # =====================================================
    # 骨盤中央
    # =====================================================

    metrics[
        "pelvis_mid_x"
    ] = (
        lh.x
        + rh.x
    ) / 2

    metrics[
        "pelvis_mid_y"
    ] = (
        lh.y
        + rh.y
    ) / 2

    metrics[
        "pelvis_mid_z"
    ] = (
        lh.z
        + rh.z
    ) / 2


    return metrics


# =========================================================
# World座標
# =========================================================

def extract_world_metrics(
    world_landmarks,
):
    """
    MediaPipe Pose World Landmarksを保存する。

    注意：
    三次元動作解析装置による座標と
    同等には扱わない。
    """

    metrics = {}


    for (
        point_name,
        point_index,
    ) in POSE_TRAJECTORY_POINTS.items():

        landmark = (
            world_landmarks[
                point_index
            ]
        )

        metrics[
            f"{point_name}_world_x"
        ] = float(
            landmark.x
        )

        metrics[
            f"{point_name}_world_y"
        ] = float(
            landmark.y
        )

        metrics[
            f"{point_name}_world_z"
        ] = float(
            landmark.z
        )


    # 肩中央
    ls = world_landmarks[
        POSE_INDEX[
            "left_shoulder"
        ]
    ]

    rs = world_landmarks[
        POSE_INDEX[
            "right_shoulder"
        ]
    ]

    metrics[
        "shoulder_mid_world_x"
    ] = (
        ls.x
        + rs.x
    ) / 2

    metrics[
        "shoulder_mid_world_y"
    ] = (
        ls.y
        + rs.y
    ) / 2

    metrics[
        "shoulder_mid_world_z"
    ] = (
        ls.z
        + rs.z
    ) / 2


    # 骨盤中央
    lh = world_landmarks[
        POSE_INDEX[
            "left_hip"
        ]
    ]

    rh = world_landmarks[
        POSE_INDEX[
            "right_hip"
        ]
    ]

    metrics[
        "pelvis_mid_world_x"
    ] = (
        lh.x
        + rh.x
    ) / 2

    metrics[
        "pelvis_mid_world_y"
    ] = (
        lh.y
        + rh.y
    ) / 2

    metrics[
        "pelvis_mid_world_z"
    ] = (
        lh.z
        + rh.z
    ) / 2


    return metrics


# =========================================================
# 手指軌跡初期値
# =========================================================

def create_empty_hand_metrics():
    """
    手が検出されないフレームでも
    列構造を保持するための初期値。
    """

    metrics = {
        "left_pinch_norm": np.nan,
        "right_pinch_norm": np.nan,
    }


    for side in [
        "left",
        "right",
    ]:

        for point_name in (
            HAND_POINT_INDICES.keys()
        ):

            full_name = (
                f"{side}_"
                f"{point_name}"
            )

            metrics[
                f"{full_name}_x"
            ] = np.nan

            metrics[
                f"{full_name}_y"
            ] = np.nan

            metrics[
                f"{full_name}_z"
            ] = np.nan


    return metrics


# =========================================================
# 手指数値抽出
# =========================================================

def add_hand_metrics(
    row,
    hand_landmarks,
    side,
):
    """
    左右の手について、

    ・ピンチ距離
    ・手関節
    ・各指先

    の座標を保存する。
    """

    pinch = (
        calculate_pinch_distance(
            hand_landmarks
        )
    )

    row[
        f"{side}_pinch_norm"
    ] = pinch


    for (
        point_name,
        point_index,
    ) in HAND_POINT_INDICES.items():

        landmark = (
            hand_landmarks[
                point_index
            ]
        )

        full_name = (
            f"{side}_"
            f"{point_name}"
        )

        row[
            f"{full_name}_x"
        ] = float(
            landmark.x
        )

        row[
            f"{full_name}_y"
        ] = float(
            landmark.y
        )

        row[
            f"{full_name}_z"
        ] = float(
            landmark.z
        )


# =========================================================
# 全軌跡点一覧
# =========================================================

def get_all_trajectory_point_names():
    """
    数値軌跡として扱う点の一覧。
    """

    return list(
        TRAJECTORY_POINT_LABELS.keys()
    )


# =========================================================
# 移動距離・速度追加
# =========================================================

def add_trajectory_kinematics(
    df,
):
    """
    各軌跡点について、

    ・フレーム間移動距離
    ・速度

    を2D正規化座標から計算する。
    """

    if df.empty:

        return df


    delta_time = (
        df["time_s"]
        .diff()
        .replace(
            0,
            np.nan,
        )
    )


    for point_name in (
        get_all_trajectory_point_names()
    ):

        x_column = (
            f"{point_name}_x"
        )

        y_column = (
            f"{point_name}_y"
        )


        if (
            x_column
            not in df.columns
            or
            y_column
            not in df.columns
        ):

            continue


        dx = (
            df[x_column]
            .diff()
        )

        dy = (
            df[y_column]
            .diff()
        )


        distance = np.sqrt(
            dx ** 2
            + dy ** 2
        )


        df[
            f"{point_name}_distance_norm"
        ] = distance


        df[
            f"{point_name}_speed_norm_s"
        ] = (
            distance
            / delta_time
        )


    return df


# =========================================================
# 動画解析
# =========================================================

def analyze_video(
    video_path,
    phases=None,
    target_fps=30,
):
    """
    Pose + Hand Landmarkerで動画を解析する。

    戻り値：

    1. フレーム別DataFrame
    2. 動画情報
    """

    if phases is None:
        phases = []


    # =====================================================
    # モデル確認
    # =====================================================

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


    source_fps = float(
        video_info[
            "fps"
        ]
    )


    target_fps = min(
        float(
            target_fps
        ),
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
            "動画を開けませんでした。"
        )


    rows = []


    try:

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

                success, frame = (
                    cap.read()
                )


                if not success:
                    break


                # =====================================
                # フレーム間引き
                # =====================================

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


                # =====================================
                # RGB変換
                # =====================================

                rgb_frame = (
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )
                )


                mp_image = (
                    mp.Image(
                        image_format=(
                            mp.ImageFormat.SRGB
                        ),
                        data=rgb_frame,
                    )
                )


                # =====================================
                # Pose
                # =====================================

                pose_result = (
                    pose_landmarker
                    .detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )
                )


                # =====================================
                # Hand
                # =====================================

                hand_result = (
                    hand_landmarker
                    .detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )
                )


                # =====================================
                # 基本行
                # =====================================

                row = {
                    "frame":
                        frame_index,

                    "time_s":
                        time_seconds,

                    "phase":
                        get_phase(
                            time_seconds,
                            phases,
                        ),
                }


                # 手指列を先に確保
                row.update(
                    create_empty_hand_metrics()
                )


                # =====================================
                # Pose
                # =====================================

                if (
                    pose_result
                    .pose_landmarks
                ):

                    landmarks = (
                        pose_result
                        .pose_landmarks[0]
                    )


                    row.update(
                        extract_pose_metrics(
                            landmarks
                        )
                    )


                # =====================================
                # World Landmarks
                # =====================================

                if (
                    pose_result
                    .pose_world_landmarks
                ):

                    world_landmarks = (
                        pose_result
                        .pose_world_landmarks[0]
                    )


                    row.update(
                        extract_world_metrics(
                            world_landmarks
                        )
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


                        hand_side = (
                            handedness[0]
                            .category_name
                            .lower()
                        )


                        if hand_side not in [
                            "left",
                            "right",
                        ]:

                            continue


                        add_hand_metrics(
                            row,
                            hand_landmarks,
                            hand_side,
                        )


                rows.append(
                    row
                )


                frame_index += 1


    finally:

        cap.release()


    # =====================================================
    # DataFrame
    # =====================================================

    df = pd.DataFrame(
        rows
    )


    if df.empty:

        raise ValueError(
            "解析可能なフレームがありませんでした。"
        )


    # =====================================================
    # 軌跡距離・速度
    # =====================================================

    df = (
        add_trajectory_kinematics(
            df
        )
    )


    return (
        df,
        video_info,
    )


# =========================================================
# 1軌跡の工程別集計
# =========================================================

def calculate_phase_trajectory_metrics(
    group,
    point_name,
):
    """
    1つの軌跡点について工程内の、

    ・総移動距離
    ・開始→終了変位
    ・平均速度
    ・最大速度
    ・左右方向移動範囲
    ・上下方向移動範囲

    を算出する。
    """

    x_column = (
        f"{point_name}_x"
    )

    y_column = (
        f"{point_name}_y"
    )


    if (
        x_column
        not in group.columns
        or
        y_column
        not in group.columns
    ):

        return {}


    valid = (
        group[
            [
                x_column,
                y_column,
            ]
        ]
        .dropna()
    )


    if valid.empty:

        return {}


    x = valid[
        x_column
    ]

    y = valid[
        y_column
    ]


    dx = x.diff()
    dy = y.diff()


    total_distance = (
        np.sqrt(
            dx ** 2
            + dy ** 2
        )
        .sum()
    )


    start_x = x.iloc[0]
    start_y = y.iloc[0]

    end_x = x.iloc[-1]
    end_y = y.iloc[-1]


    displacement = np.sqrt(
        (
            end_x
            - start_x
        ) ** 2
        +
        (
            end_y
            - start_y
        ) ** 2
    )


    speed_column = (
        f"{point_name}"
        "_speed_norm_s"
    )


    if (
        speed_column
        in group.columns
    ):

        mean_speed = (
            group[
                speed_column
            ]
            .mean()
        )

        max_speed = (
            group[
                speed_column
            ]
            .max()
        )

    else:

        mean_speed = np.nan
        max_speed = np.nan


    return {

        f"{point_name}_total_distance_norm":
            float(
                total_distance
            ),

        f"{point_name}_displacement_norm":
            float(
                displacement
            ),

        f"{point_name}_mean_speed_norm_s":
            float(
                mean_speed
            )
            if not np.isnan(
                mean_speed
            )
            else np.nan,

        f"{point_name}_max_speed_norm_s":
            float(
                max_speed
            )
            if not np.isnan(
                max_speed
            )
            else np.nan,

        f"{point_name}_x_range_norm":
            float(
                x.max()
                - x.min()
            ),

        f"{point_name}_y_range_norm":
            float(
                y.max()
                - y.min()
            ),
    }


# =========================================================
# 工程別集計
# =========================================================

def summarize_by_phase(
    df,
):
    """
    各工程について、

    ・時間
    ・主要2D投影角
    ・体幹傾斜
    ・全身軌跡
    ・手指軌跡

    を集計する。
    """

    if (
        "phase"
        not in df.columns
    ):

        return pd.DataFrame()


    summary_rows = []


    for (
        phase_name,
        group,
    ) in df.groupby(
        "phase",
        sort=False,
    ):


        if (
            phase_name
            == "工程外"
        ):

            continue


        row = {
            "工程":
                phase_name,

            "開始秒":
                group[
                    "time_s"
                ].min(),

            "終了秒":
                group[
                    "time_s"
                ].max(),

            "解析フレーム数":
                len(
                    group
                ),
        }


        # =================================================
        # 角度
        # =================================================

        angle_columns = [
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

            "left_pinch_norm",
            "right_pinch_norm",
        ]


        for column in (
            angle_columns
        ):

            if (
                column
                not in group.columns
            ):

                continue


            row[
                f"{column}_mean"
            ] = (
                group[
                    column
                ]
                .mean()
            )


            row[
                f"{column}_max"
            ] = (
                group[
                    column
                ]
                .max()
            )


            row[
                f"{column}_min"
            ] = (
                group[
                    column
                ]
                .min()
            )


        # =================================================
        # 全身＋手指軌跡
        # =================================================

        for point_name in (
            get_all_trajectory_point_names()
        ):

            trajectory_metrics = (
                calculate_phase_trajectory_metrics(
                    group,
                    point_name,
                )
            )

            row.update(
                trajectory_metrics
            )


        summary_rows.append(
            row
        )


    return pd.DataFrame(
        summary_rows
    )


# =========================================================
# グラフ用軌跡DataFrame
# =========================================================

def get_trajectory_dataframe(
    df,
    point_names,
):
    """
    選択した関節・部位のX/Y座標を
    グラフ表示用に抽出する。
    """

    columns = [
        "time_s",
    ]


    for point_name in (
        point_names
    ):

        for axis in [
            "x",
            "y",
        ]:

            column = (
                f"{point_name}_{axis}"
            )

            if (
                column
                in df.columns
            ):

                columns.append(
                    column
                )


    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]


    return df[
        available_columns
    ].copy()