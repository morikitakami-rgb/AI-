from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg


# =========================================================
# OT Practice Copilot
# 動画読み込み・変換ユーティリティ
# =========================================================


def can_opencv_read_video(
    video_path,
):
    """
    OpenCVが動画を正常に開けるか確認する。
    """

    video_path = Path(
        video_path
    )

    if not video_path.exists():
        return False

    if not video_path.is_file():
        return False

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    opened = cap.isOpened()

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    cap.release()

    return (
        opened
        and fps > 0
        and frame_count > 0
    )


# =========================================================
# H.264 MP4変換
# =========================================================

def convert_to_h264_mp4(
    input_path,
    output_path,
):
    """
    MOV・HEVCなどの動画を、
    OpenCV・MediaPipe・Streamlitで
    扱いやすいH.264 / MP4へ変換する。
    """

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )


    # -----------------------------------------------------
    # 入力確認
    # -----------------------------------------------------

    if not input_path.exists():

        raise FileNotFoundError(
            "入力動画が見つかりません。\n"
            f"{input_path}"
        )


    if not input_path.is_file():

        raise ValueError(
            "指定された場所は"
            "動画ファイルではありません。\n"
            f"{input_path}"
        )


    # -----------------------------------------------------
    # 出力フォルダ
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------
    # FFmpeg
    # -----------------------------------------------------

    ffmpeg = (
        imageio_ffmpeg
        .get_ffmpeg_exe()
    )


    command = [

        ffmpeg,

        "-y",

        "-i",
        str(
            input_path
        ),

        # H.264
        "-c:v",
        "libx264",

        # ブラウザ/OpenCV互換性
        "-pix_fmt",
        "yuv420p",

        # 画質
        "-crf",
        "20",

        # 速度
        "-preset",
        "fast",

        # Streamlit等で再生しやすくする
        "-movflags",
        "+faststart",

        # 動作解析には音声不要
        "-an",

        str(
            output_path
        ),
    ]


    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )


    if result.returncode != 0:

        raise RuntimeError(
            "動画をH.264 MP4へ"
            "変換できませんでした。\n\n"
            + result.stderr[-2000:]
        )


    if not output_path.exists():

        raise RuntimeError(
            "変換処理は終了しましたが、"
            "出力動画が作成されていません。"
        )


    if output_path.stat().st_size == 0:

        raise RuntimeError(
            "変換後の動画サイズが0です。"
        )


    return output_path


# =========================================================
# 解析用動画の準備
# =========================================================

def prepare_video_for_analysis(
    input_path,
    output_path=None,
    force_convert=False,
):
    """
    動画をMediaPipe解析可能な状態へ準備する。

    OpenCVで正常に読める場合：
        元動画をそのまま利用。

    MOV / HEVCなどで読めない場合：
        H.264 MP4へ自動変換する。

    force_convert=Trueの場合：
        読める動画でもMP4へ変換する。
    """

    input_path = Path(
        input_path
    )


    if not input_path.exists():

        raise FileNotFoundError(
            "動画が見つかりません。\n"
            f"{input_path}"
        )


    if not input_path.is_file():

        raise ValueError(
            "指定された場所は"
            "動画ファイルではありません。\n"
            f"{input_path}"
        )


    # -----------------------------------------------------
    # そのまま解析可能
    # -----------------------------------------------------

    if (
        not force_convert
        and can_opencv_read_video(
            input_path
        )
    ):

        return {
            "path": input_path,
            "converted": False,
        }


    # -----------------------------------------------------
    # 出力先
    # -----------------------------------------------------

    if output_path is None:

        output_path = (
            input_path.parent
            / (
                input_path.stem
                + "_converted.mp4"
            )
        )


    output_path = Path(
        output_path
    )


    # -----------------------------------------------------
    # 変換
    # -----------------------------------------------------

    converted_path = (
        convert_to_h264_mp4(
            input_path=input_path,
            output_path=output_path,
        )
    )


    # -----------------------------------------------------
    # 変換後にOpenCV確認
    # -----------------------------------------------------

    if not can_opencv_read_video(
        converted_path
    ):

        raise RuntimeError(
            "MP4への変換は完了しましたが、"
            "OpenCVで正常に読み込めませんでした。"
        )


    return {
        "path": converted_path,
        "converted": True,
    }