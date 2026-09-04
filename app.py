import json
import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from agents import Runner, SQLiteSession

from agent_core import (
    create_agent,
    build_question,
)

from practice_core import (
    PRACTICE_SCOPE_OPTIONS,
    PRACTICE_CONTEXT_OPTIONS,
    build_practice_prompt,
    build_practice_refinement_prompt,
    parse_practice_response,
)

from evidence_core import (
    run_evidence_search,
)

from research_core import (
    run_research_runner,
)

from prompts import (
    CATEGORY_MODES,
    get_mode_instruction,
)

from privacy import (
    has_sensitive_information,
    get_privacy_warning,
    PRIVACY_GUIDE,
)

from exam_mode import (
    EXAM_CATEGORIES,
    DIFFICULTY_LEVELS,
    QUESTION_COUNTS,
    EXPLANATION_LEVELS,
    build_question_prompt,
    build_explanation_prompt,
    build_case_question_prompt,
    build_study_prompt,
    build_mock_exam_prompt,
)

from presentation_builder import (
    create_presentation,
)

from motion_analysis import (
    TRAJECTORY_PRESETS,
    analyze_video,
    get_trajectory_points,
    get_video_info,
    summarize_by_phase,
)

from motion_overlay import (
    create_annotated_video,
)

from video_utils import (
    prepare_video_for_analysis,
)


# =========================================================
# Streamlit 基本設定
# =========================================================

st.set_page_config(
    page_title="OT Practice Copilot",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# 領域ごとの説明
# =========================================================

CATEGORY_DESCRIPTIONS = {

    "🩺 臨床実践支援":
        "症例、ADL・IADL、評価、目標設定、"
        "介入、環境調整、再評価などの"
        "臨床推論を支援します。",

    "💼 働くことの支援":
        "就労支援、復職、テレワーク、職場調整、"
        "疲労・ペーシング、産業保健などを"
        "支援します。",

    "🔎 エビデンス・研究支援":
        "エビデンス検索、研究計画、研究デザイン、"
        "評価指標、統計解析、倫理審査、"
        "論文作成などを支援します。",

    "🎤 学会・学術発信支援":
        "演題名、抄録、学会発表、ポスター、"
        "発表原稿、想定質問、PowerPoint作成などを"
        "支援します。",

    "🎓 教育・研修・国家試験支援":
        "大学講義、研修会、教材作成、演習、"
        "国家試験対策などを支援します。",

    "📹 動作解析Lab β":
        "動画から全身・上肢・下肢・手指の"
        "動作特徴を数値化し、"
        "作業療法の視点から整理します。",
}


# =========================================================
# Agent
# =========================================================

@st.cache_resource
def get_agent():
    """
    OT Practice Copilot Agentを作成する。
    """

    return create_agent()


try:

    agent = get_agent()

except Exception as e:

    st.error(
        "AIエージェントの初期化に失敗しました。"
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# Session State
# =========================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = (
        str(
            uuid.uuid4()
        )
    )


if "agent_session" not in st.session_state:

    st.session_state.agent_session = (
        SQLiteSession(
            st.session_state.session_id
        )
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


if "motion_df" not in st.session_state:

    st.session_state.motion_df = None


if "motion_summary" not in st.session_state:

    st.session_state.motion_summary = None


if "motion_overlay_bytes" not in st.session_state:

    st.session_state.motion_overlay_bytes = None


if "motion_overlay_filename" not in st.session_state:

    st.session_state.motion_overlay_filename = None


if "motion_source_key" not in st.session_state:

    st.session_state.motion_source_key = None


if "motion_preset" not in st.session_state:

    st.session_state.motion_preset = "全身"


if "exam_questions" not in st.session_state:

    st.session_state.exam_questions = None


if "pptx_bytes" not in st.session_state:

    st.session_state.pptx_bytes = None


if "pptx_filename" not in st.session_state:

    st.session_state.pptx_filename = None


if "practice_result" not in st.session_state:

    st.session_state.practice_result = None


if "practice_question" not in st.session_state:

    st.session_state.practice_question = ""

if "practice_evidence_result" not in st.session_state:

    st.session_state.practice_evidence_result = None

if "practice_history" not in st.session_state:

    st.session_state.practice_history = []

if "research_runner_result" not in st.session_state:

    st.session_state.research_runner_result = None

# =========================================================
# AI実行
# =========================================================

def run_agent(
    prompt,
    use_conversation=True,
):
    """
    OpenAI Agentを実行する。

    use_conversation=True:
        現在の会話文脈を保持する。

    use_conversation=False:
        独立した1回の処理として実行する。
    """

    if use_conversation:

        result = Runner.run_sync(
            agent,
            prompt,
            session=(
                st.session_state
                .agent_session
            ),
        )

    else:

        result = Runner.run_sync(
            agent,
            prompt,
        )

    return result.final_output


# =========================================================
# PowerPoint用JSON抽出
# =========================================================

def extract_json_array(text):
    """
    AI出力からJSON配列を抽出する。
    """

    text = text.strip()

    if text.startswith("```"):

        text = text.replace(
            "```json",
            "",
            1,
        )

        text = text.replace(
            "```",
            "",
        )

    start = text.find("[")
    end = text.rfind("]")

    if (
        start == -1
        or end == -1
    ):

        raise ValueError(
            "PowerPoint用のJSONを"
            "取得できませんでした。"
        )

    return json.loads(
        text[
            start:end + 1
        ]
    )


# =========================================================
# 新しい会話
# =========================================================

def new_conversation():
    """
    現在の会話履歴を破棄し、
    新しいインメモリ会話を開始する。
    """

    st.session_state.session_id = (
        str(
            uuid.uuid4()
        )
    )

    st.session_state.agent_session = (
        SQLiteSession(
            st.session_state.session_id
        )
    )

    st.session_state.messages = []

    st.session_state.exam_questions = None

    st.session_state.motion_df = None

    st.session_state.motion_summary = None

    st.session_state.motion_overlay_bytes = None

    st.session_state.motion_overlay_filename = None

    st.session_state.motion_source_key = None

    st.rerun()


# =========================================================
# サイドバー
# =========================================================

with st.sidebar:

    st.header(
        "🧠 OT Practice Copilot"
    )

    st.caption(
        "作業療法の実践・研究・教育を支援"
    )

    st.divider()


    # -----------------------------------------------------
    # 大領域
    # -----------------------------------------------------

    category = st.selectbox(
        "利用する領域",
        list(
            CATEGORY_MODES.keys()
        ),
        key="category_selector",
    )


    # -----------------------------------------------------
    # モード
    # -----------------------------------------------------

    modes = CATEGORY_MODES[
        category
    ]

    mode = st.selectbox(
        "内容を選択",
        modes,
        key="mode_selector",
    )


    st.divider()


    # -----------------------------------------------------
    # 匿名化確認
    # -----------------------------------------------------

    privacy_confirmed = (
        st.checkbox(
            "個人を特定できる情報を入力しません",
            value=False,
        )
    )

    st.caption(
        "臨床・就労・教育等では、"
        "必要最小限の匿名化情報のみを"
        "使用してください。"
    )


    st.divider()


    # -----------------------------------------------------
    # 新しい会話
    # -----------------------------------------------------

    if st.button(
        "🔄 新しい会話",
        use_container_width=True,
    ):

        new_conversation()


# =========================================================
# アプリ全体のヘッダー
# =========================================================

st.title(
    "🧠 OT Practice Copilot"
)

st.caption(
    "作業療法士・研究者・教員・学生の"
    "臨床・就労・研究・教育・学術活動を"
    "支援するAI Copilot"
)


# =========================================================
# 現在選択している領域を明確に表示
# =========================================================

st.markdown(
    f"## {category}"
)

st.write(
    CATEGORY_DESCRIPTIONS.get(
        category,
        ""
    )
)

st.markdown(
    f"### 📌 {mode}"
)

st.caption(
    "選択した領域・モードに応じて、"
    "AIへの専門的な指示も切り替わります。"
)


# =========================================================
# 匿名化ガイド
# =========================================================

with st.expander(
    "🔒 個人情報・匿名化について",
    expanded=False,
):

    st.markdown(
        PRIVACY_GUIDE
    )


st.divider()


# =========================================================
# PowerPoint作成
# =========================================================

if mode in [
    "PowerPoint作成",
    "講義PowerPoint作成",
]:

    st.subheader(
        "📊 PowerPoint作成 β"
    )

    st.caption(
        "AIがスライド構成を作成し、"
        "PowerPoint（.pptx）として出力します。"
    )


    # -----------------------------------------------------
    # 資料の種類
    # -----------------------------------------------------

    if mode == "PowerPoint作成":

        presentation_type = (
            "学会発表"
        )

    else:

        presentation_type = (
            st.selectbox(
                "資料の種類",
                [
                    "大学講義",
                    "研修会・セミナー",
                ],
            )
        )


    # -----------------------------------------------------
    # 入力項目
    # -----------------------------------------------------

    presentation_title = (
        st.text_input(
            "タイトル",
            placeholder=(
                "例：作業療法士による就労支援"
            ),
        )
    )


    audience = st.text_input(
        "対象者",
        placeholder=(
            "例：臨床経験5年以上の作業療法士"
        ),
    )


    duration = st.number_input(
        "発表・講義時間（分）",
        min_value=5,
        max_value=300,
        value=60,
        step=5,
    )


    slide_count = st.number_input(
        "本文スライド枚数",
        min_value=3,
        max_value=80,
        value=15,
        step=1,
    )


    objective = st.text_area(
        "目的・到達目標",
        height=100,
        placeholder=(
            "この発表・講義で"
            "何を理解してほしいか入力してください。"
        ),
    )


    additional_info = st.text_area(
        "含めたい内容・条件",
        height=150,
        placeholder=(
            "例：事例を1つ入れる、"
            "最後に演習を入れる、"
            "最新エビデンスを示す"
        ),
    )


    # -----------------------------------------------------
    # PowerPoint生成
    # -----------------------------------------------------

    if st.button(
        "✨ スライド構成とPowerPointを作成",
        type="primary",
    ):

        if not privacy_confirmed:

            st.warning(
                "個人を特定できる情報を"
                "入力していないことを確認してください。"
            )

        elif not presentation_title:

            st.warning(
                "タイトルを入力してください。"
            )

        else:

            ppt_prompt = f"""
{presentation_type}のPowerPointを作成します。

【タイトル】
{presentation_title}

【対象者】
{audience}

【時間】
{duration}分

【本文スライド枚数】
{slide_count}枚

【目的・到達目標】
{objective}

【追加条件】
{additional_info}


以下のJSON配列だけを出力してください。

Markdown記法や説明文は不要です。


[
  {{
    "title": "スライドタイトル",
    "bullets": [
      "要点1",
      "要点2"
    ],
    "speaker_notes":
      "このスライドで講師または発表者が話す内容"
  }}
]


【作成条件】

・1枚1メッセージを基本とする

・文章を詰め込みすぎない

・対象者の知識レベルを考慮する

・指定された時間に収まる構成とする

・指定された本文スライド枚数に近づける

・スライド本文だけではなく、
  講師・発表者ノートも作成する

・学会発表の場合は、
  背景、目的、方法、結果、考察、結論の
  論理的一貫性を重視する

・大学講義の場合は、
  到達目標、説明、事例、確認、まとめを
  考慮する

・研修会の場合は、
  講義、事例、演習、実践への応用を
  必要に応じて含める

・エビデンスが必要な内容は
  可能な限り確認する

・存在しない論文や文献を作らない
"""

            try:

                with st.spinner(
                    "スライド構成を作成しています..."
                ):

                    ai_output = (
                        run_agent(
                            ppt_prompt,
                            use_conversation=False,
                        )
                    )

                    slides = (
                        extract_json_array(
                            ai_output
                        )
                    )


                # -----------------------------------------
                # 一時PowerPointファイル作成
                # -----------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pptx",
                ) as tmp:

                    temp_path = (
                        tmp.name
                    )


                create_presentation(
                    slides=slides,
                    output_path=temp_path,
                    presentation_title=(
                        presentation_title
                    ),
                    presentation_type=(
                        presentation_type
                    ),
                    subtitle=(
                        "Generated with "
                        "OT Practice Copilot"
                    ),
                )


                # -----------------------------------------
                # ファイル読み込み
                # -----------------------------------------

                with open(
                    temp_path,
                    "rb",
                ) as file:

                    st.session_state.pptx_bytes = (
                        file.read()
                    )


                os.remove(
                    temp_path
                )


                st.session_state.pptx_filename = (
                    "ot_practice_copilot.pptx"
                )


                st.success(
                    "PowerPointを作成しました。"
                )


                # -----------------------------------------
                # スライド構成表示
                # -----------------------------------------

                st.markdown(
                    "### 作成したスライド構成"
                )


                for index, slide in enumerate(
                    slides,
                    start=1,
                ):

                    slide_title = (
                        slide.get(
                            "title",
                            "",
                        )
                    )


                    with st.expander(
                        f"{index}. {slide_title}"
                    ):

                        bullets = slide.get(
                            "bullets",
                            [],
                        )


                        for bullet in bullets:

                            if isinstance(
                                bullet,
                                dict,
                            ):

                                st.markdown(
                                    f"- "
                                    f"{bullet.get('text', '')}"
                                )

                            else:

                                st.markdown(
                                    f"- {bullet}"
                                )


                        notes = slide.get(
                            "speaker_notes",
                            "",
                        )


                        if notes:

                            st.markdown(
                                "**講師・発表者ノート**"
                            )

                            st.write(
                                notes
                            )


            except Exception as e:

                st.error(
                    "PowerPoint作成中に"
                    "エラーが発生しました。"
                )

                st.code(
                    str(e)
                )


    # -----------------------------------------------------
    # PowerPointダウンロード
    # -----------------------------------------------------

    if (
        st.session_state.pptx_bytes
        is not None
    ):

        st.download_button(
            "📥 PowerPointをダウンロード",
            data=(
                st.session_state
                .pptx_bytes
            ),
            file_name=(
                st.session_state
                .pptx_filename
            ),
            mime=(
                "application/"
                "vnd.openxmlformats-officedocument."
                "presentationml.presentation"
            ),
        )


# =========================================================
# 動作解析Lab
# =========================================================

elif category == "📹 動作解析Lab β":

    st.warning(
        "本機能で算出する角度は、"
        "単眼動画から得られる2D投影値です。"
        "ゴニオメーターや三次元動作解析装置による"
        "正式な関節角度測定の代替ではありません。"
    )


    st.info(
        "元動画はPC内でMediaPipe / OpenCVにより解析します。"
        "元動画そのものをOpenAIへ送信しません。"
        "AIによる整理を実行した場合に送信するのは、"
        "解析後の匿名化された数値情報です。"
    )


    st.caption(
        "MOV・HEVCなどOpenCVで直接読み込みにくい動画は、"
        "解析時に一時的なH.264 / MP4へ自動変換します。"
        "一時ファイルは処理終了後に削除します。"
    )


    # -----------------------------------------------------
    # 動画確認
    # -----------------------------------------------------

    video_confirmed = st.checkbox(
        "テスト用または適切に匿名化された"
        "動画であることを確認しました",
        key="motion_video_confirmed",
    )


    uploaded_video = st.file_uploader(
        "解析する動画",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
        ],
        key="motion_video_uploader",
    )


    # -----------------------------------------------------
    # 動画が選択された場合
    # -----------------------------------------------------

    if uploaded_video:

        source_key = (
            f"{uploaded_video.name}:"
            f"{uploaded_video.size}"
        )


        # 新しい動画が選択された場合は、
        # 以前の解析結果を画面に残さない。
        if (
            st.session_state.motion_source_key
            != source_key
        ):

            st.session_state.motion_source_key = (
                source_key
            )

            st.session_state.motion_df = None

            st.session_state.motion_summary = None

            st.session_state.motion_overlay_bytes = (
                None
            )

            st.session_state.motion_overlay_filename = (
                None
            )


        st.subheader(
            "🎞️ 元動画"
        )


        st.video(
            uploaded_video.getvalue()
        )


        suffix = (
            Path(
                uploaded_video.name
            )
            .suffix
            .lower()
        )


        if not suffix:

            suffix = ".mp4"


        # =============================================
        # 動画情報を安全に取得
        # =============================================

        inspect_path = None
        inspect_converted_path = None
        video_info = None
        inspection_converted = False


        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:

                tmp.write(
                    uploaded_video.getvalue()
                )

                inspect_path = (
                    tmp.name
                )


            inspect_converted_path = (
                inspect_path
                + "_converted.mp4"
            )


            with st.spinner(
                "動画形式を確認しています..."
            ):

                prepared = (
                    prepare_video_for_analysis(
                        input_path=inspect_path,
                        output_path=(
                            inspect_converted_path
                        ),
                    )
                )


                inspection_converted = (
                    prepared[
                        "converted"
                    ]
                )


                prepared_path = str(
                    prepared[
                        "path"
                    ]
                )


                video_info = (
                    get_video_info(
                        prepared_path
                    )
                )


        except Exception as e:

            st.error(
                "動画を読み込めませんでした。"
            )

            st.code(
                str(e)
            )


        finally:

            for temp_file in [
                inspect_path,
                inspect_converted_path,
            ]:

                if (
                    temp_file
                    and os.path.exists(
                        temp_file
                    )
                ):

                    try:

                        os.remove(
                            temp_file
                        )

                    except OSError:

                        pass


        if video_info is None:

            st.stop()


        if inspection_converted:

            st.success(
                "この動画は解析可能なH.264 / MP4へ"
                "一時的に自動変換して処理できます。"
            )


        # -----------------------------------------
        # 動画基本情報
        # -----------------------------------------

        c1, c2, c3 = st.columns(3)


        c1.metric(
            "FPS",
            f"{video_info['fps']:.1f}",
        )


        c2.metric(
            "動画時間",
            f"{video_info['duration']:.2f}秒",
        )


        c3.metric(
            "フレーム数",
            int(
                video_info[
                    "frame_count"
                ]
            ),
        )


        st.divider()


        # =========================================
        # 解析設定
        # =========================================

        st.subheader(
            "⚙️ 解析設定"
        )


        setting_col1, setting_col2 = (
            st.columns(2)
        )


        preset_options = list(
            TRAJECTORY_PRESETS.keys()
        )


        if "全身" in preset_options:

            default_preset_index = (
                preset_options.index(
                    "全身"
                )
            )

        else:

            default_preset_index = 0


        with setting_col1:

            trajectory_preset = (
                st.selectbox(
                    "軌跡プリセット",
                    preset_options,
                    index=(
                        default_preset_index
                    ),
                    help=(
                        "動画上に軌跡を表示する"
                        "身体部位の組み合わせです。"
                    ),
                )
            )


            trajectory_duration_s = (
                st.selectbox(
                    "軌跡を残す時間",
                    [
                        0.5,
                        1.5,
                        3.0,
                    ],
                    index=1,
                    format_func=(
                        lambda value:
                        f"{value:.1f}秒"
                    ),
                )
            )


            analysis_fps = (
                st.selectbox(
                    "数値解析の頻度",
                    [
                        10,
                        15,
                        30,
                    ],
                    index=2,
                    help=(
                        "角度・座標などの"
                        "数値データを解析する頻度です。"
                    ),
                )
            )


        with setting_col2:

            show_skeleton = (
                st.checkbox(
                    "骨格を表示",
                    value=True,
                )
            )


            show_angles = (
                st.checkbox(
                    "2D投影角を表示",
                    value=True,
                )
            )


            show_trajectory = (
                st.checkbox(
                    "軌跡を表示",
                    value=True,
                )
            )


            show_hands = (
                st.checkbox(
                    "手指ランドマークを表示",
                    value=True,
                )
            )


            show_trajectory_labels = (
                st.checkbox(
                    "軌跡ラベルを表示",
                    value=False,
                    help=(
                        "全身プリセットでは"
                        "表示が混雑しやすいため、"
                        "通常はOFFを推奨します。"
                    ),
                )
            )


            reset_trajectory_each_phase = (
                st.checkbox(
                    "工程が変わるたびに軌跡をリセット",
                    value=False,
                )
            )


        selected_trajectory_points = (
            get_trajectory_points(
                trajectory_preset
            )
        )


        with st.expander(
            "今回追跡する身体部位",
            expanded=False,
        ):

            if selected_trajectory_points:

                st.write(
                    "、".join(
                        selected_trajectory_points
                    )
                )

            else:

                st.write(
                    "追跡対象が設定されていません。"
                )


        # =========================================
        # 工程設定
        # =========================================

        st.subheader(
            "🧩 工程設定"
        )


        st.caption(
            "動画を作業工程ごとに区切ると、"
            "工程別の角度・軌跡・速度の特徴を"
            "比較しやすくなります。"
        )


        phase_count = int(
            st.number_input(
                "工程数",
                min_value=1,
                max_value=12,
                value=4,
                step=1,
            )
        )


        phases = []


        duration_seconds = float(
            video_info[
                "duration"
            ]
        )


        for i in range(
            phase_count
        ):

            default_start = (
                duration_seconds
                * i
                / phase_count
            )


            default_end = (
                duration_seconds
                * (i + 1)
                / phase_count
            )


            col1, col2, col3 = (
                st.columns(
                    [
                        2,
                        1,
                        1,
                    ]
                )
            )


            phase_name = (
                col1.text_input(
                    f"工程 {i + 1}",
                    value=(
                        f"工程{i + 1}"
                    ),
                    key=(
                        f"phase_name_{i}"
                    ),
                )
            )


            start_time = (
                col2.number_input(
                    "開始秒",
                    min_value=0.0,
                    max_value=(
                        duration_seconds
                    ),
                    value=float(
                        default_start
                    ),
                    step=0.1,
                    key=(
                        f"phase_start_{i}"
                    ),
                )
            )


            end_time = (
                col3.number_input(
                    "終了秒",
                    min_value=0.0,
                    max_value=(
                        duration_seconds
                    ),
                    value=float(
                        default_end
                    ),
                    step=0.1,
                    key=(
                        f"phase_end_{i}"
                    ),
                )
            )


            phases.append(
                {
                    "name": phase_name,
                    "start": start_time,
                    "end": end_time,
                }
            )


        # -----------------------------------------
        # 工程時刻チェック
        # -----------------------------------------

        phases_valid = all(
            phase["end"]
            > phase["start"]
            for phase in phases
        )


        if not phases_valid:

            st.error(
                "各工程の終了時刻は"
                "開始時刻より後にしてください。"
            )


        # =========================================
        # 動作解析開始
        # =========================================

        st.divider()


        if st.button(
            "📊 数値解析＋軌跡動画を作成",
            type="primary",
            disabled=(
                not video_confirmed
                or not phases_valid
            ),
            use_container_width=True,
        ):

            source_path = None
            converted_path = None
            overlay_path = None


            # 新しい解析を開始する時点で、
            # 古い結果をクリアする。
            st.session_state.motion_df = None

            st.session_state.motion_summary = None

            st.session_state.motion_overlay_bytes = (
                None
            )

            st.session_state.motion_overlay_filename = (
                None
            )


            try:

                # ---------------------------------
                # 元動画を一時ファイルへ保存
                # ---------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as tmp:

                    tmp.write(
                        uploaded_video.getvalue()
                    )

                    source_path = (
                        tmp.name
                    )


                converted_path = (
                    source_path
                    + "_analysis.mp4"
                )


                # ---------------------------------
                # MOV / HEVC等を必要に応じて変換
                # ---------------------------------

                with st.spinner(
                    "動画形式を確認し、"
                    "必要に応じてMP4へ変換しています..."
                ):

                    prepared = (
                        prepare_video_for_analysis(
                            input_path=source_path,
                            output_path=(
                                converted_path
                            ),
                        )
                    )


                    analysis_video_path = str(
                        prepared[
                            "path"
                        ]
                    )


                if prepared[
                    "converted"
                ]:

                    st.info(
                        "解析用にH.264 / MP4へ"
                        "一時変換しました。"
                    )


                # ---------------------------------
                # 数値解析
                # ---------------------------------

                with st.spinner(
                    "MediaPipeで全身・手指の"
                    "数値解析をしています..."
                ):

                    (
                        motion_df,
                        _
                    ) = analyze_video(
                        video_path=(
                            analysis_video_path
                        ),
                        phases=phases,
                        target_fps=(
                            analysis_fps
                        ),
                    )


                    summary_df = (
                        summarize_by_phase(
                            motion_df
                        )
                    )


                # ---------------------------------
                # 骨格・角度・軌跡動画
                # ---------------------------------

                overlay_file = (
                    tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4",
                    )
                )

                overlay_path = (
                    overlay_file.name
                )

                overlay_file.close()


                # motion_overlay側で出力するため、
                # 先に作られた空ファイルを削除する。
                if os.path.exists(
                    overlay_path
                ):

                    os.remove(
                        overlay_path
                    )


                with st.spinner(
                    "骨格・関節角度・軌跡を"
                    "重ねた解析動画を作成しています..."
                ):

                    created_overlay_path = (
                        create_annotated_video(
                            video_path=(
                                analysis_video_path
                            ),
                            output_path=(
                                overlay_path
                            ),
                            phases=phases,
                            trajectory_preset=(
                                trajectory_preset
                            ),
                            trajectory_duration_s=(
                                trajectory_duration_s
                            ),
                            show_skeleton=(
                                show_skeleton
                            ),
                            show_angles=(
                                show_angles
                            ),
                            show_hands=(
                                show_hands
                            ),
                            show_trajectory=(
                                show_trajectory
                            ),
                            show_trajectory_labels=(
                                show_trajectory_labels
                            ),
                            reset_trajectory_each_phase=(
                                reset_trajectory_each_phase
                            ),
                        )
                    )


                with open(
                    created_overlay_path,
                    "rb",
                ) as video_file:

                    overlay_bytes = (
                        video_file.read()
                    )


                # ---------------------------------
                # Session Stateへ保存
                # ---------------------------------

                st.session_state.motion_df = (
                    motion_df
                )


                st.session_state.motion_summary = (
                    summary_df
                )


                st.session_state.motion_overlay_bytes = (
                    overlay_bytes
                )


                st.session_state.motion_overlay_filename = (
                    "ot_motion_analysis_overlay.mp4"
                )


                st.session_state.motion_preset = (
                    trajectory_preset
                )


                st.success(
                    "数値解析と軌跡付き解析動画の"
                    "作成が完了しました。"
                )


            except Exception as e:

                st.error(
                    "動作解析中に"
                    "エラーが発生しました。"
                )


                st.code(
                    str(e)
                )


            finally:

                for temp_file in [
                    source_path,
                    converted_path,
                    overlay_path,
                ]:

                    if (
                        temp_file
                        and os.path.exists(
                            temp_file
                        )
                    ):

                        try:

                            os.remove(
                                temp_file
                            )

                        except OSError:

                            pass


    # =====================================================
    # 動作解析結果
    # =====================================================

    if (
        uploaded_video
        and st.session_state.motion_df
        is not None
    ):

        motion_df = (
            st.session_state
            .motion_df
        )


        summary_df = (
            st.session_state
            .motion_summary
        )


        st.divider()


        st.header(
            "📊 動作解析結果"
        )


        # =============================================
        # 解析済み動画
        # =============================================

        if (
            st.session_state
            .motion_overlay_bytes
            is not None
        ):

            st.subheader(
                "🎥 骨格・角度・軌跡付き解析動画"
            )


            st.video(
                st.session_state
                .motion_overlay_bytes
            )


            st.download_button(
                "📥 解析動画をダウンロード",
                data=(
                    st.session_state
                    .motion_overlay_bytes
                ),
                file_name=(
                    st.session_state
                    .motion_overlay_filename
                    or
                    "ot_motion_analysis_overlay.mp4"
                ),
                mime="video/mp4",
            )


        # =============================================
        # 工程別解析結果
        # =============================================

        st.subheader(
            "📋 工程別解析結果"
        )


        st.dataframe(
            summary_df,
            use_container_width=True,
        )


        # =============================================
        # 角度グラフ
        # =============================================

        side = st.radio(
            "角度グラフの表示側",
            [
                "左",
                "右",
            ],
            horizontal=True,
        )


        prefix = (
            "left"
            if side == "左"
            else "right"
        )


        possible_columns = [
            f"{prefix}_shoulder_proj_deg",
            f"{prefix}_elbow_proj_deg",
            f"{prefix}_hip_proj_deg",
            f"{prefix}_knee_proj_deg",
            f"{prefix}_ankle_proj_deg",
            "trunk_tilt_deg",
        ]


        graph_columns = [
            column
            for column in possible_columns
            if column
            in motion_df.columns
        ]


        if graph_columns:

            graph_df = (
                motion_df[
                    [
                        "time_s"
                    ]
                    + graph_columns
                ]
                .set_index(
                    "time_s"
                )
            )


            st.subheader(
                "📈 時間―2D投影角変化"
            )


            st.line_chart(
                graph_df
            )


        else:

            st.info(
                "角度グラフとして表示できる"
                "列がありませんでした。"
            )


        # =============================================
        # 軌跡座標
        # =============================================

        st.subheader(
            "🧭 身体部位の軌跡・移動"
        )


        result_preset = (
            st.session_state
            .motion_preset
        )


        result_points = (
            get_trajectory_points(
                result_preset
            )
        )


        coordinate_points = []


        for point in result_points:

            x_column = (
                f"{point}_x"
            )

            y_column = (
                f"{point}_y"
            )


            if (
                x_column
                in motion_df.columns
                and y_column
                in motion_df.columns
            ):

                coordinate_points.append(
                    point
                )


        if coordinate_points:

            trajectory_point = (
                st.selectbox(
                    "表示する身体部位",
                    coordinate_points,
                    key=(
                        "trajectory_result_point"
                    ),
                )
            )


            trajectory_columns = [
                "time_s",
                f"{trajectory_point}_x",
                f"{trajectory_point}_y",
            ]


            trajectory_graph_df = (
                motion_df[
                    trajectory_columns
                ]
                .dropna()
                .set_index(
                    "time_s"
                )
            )


            if not trajectory_graph_df.empty:

                st.caption(
                    "X・Yは画像上の正規化座標です。"
                    "カメラ位置や撮影距離の影響を受けます。"
                )


                st.line_chart(
                    trajectory_graph_df
                )


            speed_columns = [
                column
                for column in motion_df.columns
                if (
                    column.startswith(
                        f"{trajectory_point}_"
                    )
                    and "speed"
                    in column.lower()
                )
            ]


            if speed_columns:

                speed_graph_df = (
                    motion_df[
                        [
                            "time_s"
                        ]
                        + speed_columns
                    ]
                    .set_index(
                        "time_s"
                    )
                )


                st.markdown(
                    "#### 移動速度"
                )


                st.line_chart(
                    speed_graph_df
                )


        else:

            st.info(
                "選択したプリセットについて、"
                "表示可能なX・Y軌跡座標が"
                "見つかりませんでした。"
            )


        # =============================================
        # データ確認
        # =============================================

        with st.expander(
            "フレーム別データを確認",
            expanded=False,
        ):

            st.dataframe(
                motion_df,
                use_container_width=True,
            )


        # =============================================
        # CSV
        # =============================================

        download_col1, download_col2 = (
            st.columns(2)
        )


        with download_col1:

            st.download_button(
                "📥 フレーム別CSV",
                data=(
                    motion_df
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8-sig"
                    )
                ),
                file_name=(
                    "motion_frame_data.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


        with download_col2:

            st.download_button(
                "📥 工程別CSV",
                data=(
                    summary_df
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8-sig"
                    )
                ),
                file_name=(
                    "motion_phase_summary.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


        # =============================================
        # AIによるOT視点の整理
        # =============================================

        st.divider()


        st.subheader(
            "🧠 OT視点による整理"
        )


        st.caption(
            "AIは元動画を見ません。"
            "PC内解析で得られた匿名化済みの"
            "工程別数値のみを使用します。"
        )


        if st.button(
            "🧠 OT視点で解析結果を整理"
        ):

            summary_text = (
                summary_df
                .round(3)
                .to_csv(
                    index=False
                )
            )


            interpretation_prompt = f"""
以下はPC内で動画解析して得られた
匿名化済みの工程別数値です。

あなた自身は元動画を見ていません。


【軌跡プリセット】

{result_preset}


【工程別データ】

{summary_text}


以下の順序で整理してください。

1. 数値として観測された特徴

2. 工程間の違い

3. 左右差

4. 作業遂行上考えられる仮説

5. 代償動作の可能性

6. 追加で確認したい評価

7. 解釈上の限界


【重要】

単眼動画から算出した
2D投影値・画像上の座標・
MediaPipeによる推定値です。

正式なROM測定値や
三次元動作解析値として
扱わないでください。

観測された事実と
AIによる仮説を
明確に区別してください。

元動画を見たかのような
表現はしないでください。
"""


            try:

                with st.spinner(
                    "解析結果をOTの視点から"
                    "整理しています..."
                ):

                    interpretation = (
                        run_agent(
                            interpretation_prompt,
                            use_conversation=False,
                        )
                    )


                st.markdown(
                    interpretation
                )


            except Exception as e:

                st.error(
                    "AIによる解釈中に"
                    "エラーが発生しました。"
                )


                st.code(
                    str(e)
                )


# =========================================================
# 国家試験対策
# =========================================================

elif (
    category
    == "🎓 教育・研修・国家試験支援"
    and mode in [
        "分野別学習",
        "国家試験問題演習",
        "症例問題",
        "苦手分野対策",
        "間違い復習",
        "ミニ模試",
    ]
):

    st.info(
        "AIが新しく作成した問題は、"
        "実際の国家試験過去問題ではありません。"
        "「AI作成オリジナル問題」として扱います。"
    )


    # -----------------------------------------------------
    # 大分類
    # -----------------------------------------------------

    main_category = (
        st.selectbox(
            "大分類",
            list(
                EXAM_CATEGORIES.keys()
            ),
        )
    )


    # -----------------------------------------------------
    # テーマ
    # -----------------------------------------------------

    topic = st.selectbox(
        "テーマ",
        EXAM_CATEGORIES[
            main_category
        ],
    )


    # -----------------------------------------------------
    # 解説レベル
    # -----------------------------------------------------

    explanation_level = (
        st.selectbox(
            "解説レベル",
            EXPLANATION_LEVELS,
        )
    )


    # =====================================================
    # 分野別学習
    # =====================================================

    if mode == "分野別学習":

        if st.button(
            "📚 学習を開始",
            type="primary",
        ):

            prompt = (
                build_study_prompt(
                    topic,
                    explanation_level,
                )
            )


            try:

                with st.spinner(
                    "学習内容を作成しています..."
                ):

                    output = (
                        run_agent(
                            prompt,
                            use_conversation=False,
                        )
                    )


                st.markdown(
                    output
                )


            except Exception as e:

                st.error(
                    "学習内容の作成中に"
                    "エラーが発生しました。"
                )

                st.code(
                    str(e)
                )


    # =====================================================
    # 国家試験問題演習
    # =====================================================

    elif mode == "国家試験問題演習":

        difficulty = (
            st.selectbox(
                "難易度",
                DIFFICULTY_LEVELS,
            )
        )


        question_count = (
            st.selectbox(
                "問題数",
                QUESTION_COUNTS,
                index=2,
            )
        )


        if st.button(
            "📝 問題を作成",
            type="primary",
        ):

            prompt = (
                build_question_prompt(
                    category=(
                        main_category
                    ),
                    topic=topic,
                    difficulty=(
                        difficulty
                    ),
                    question_count=(
                        question_count
                    ),
                    explanation_level=(
                        explanation_level
                    ),
                )
            )


            try:

                with st.spinner(
                    "問題を作成しています..."
                ):

                    st.session_state.exam_questions = (
                        run_agent(
                            prompt,
                            use_conversation=False,
                        )
                    )


            except Exception as e:

                st.error(
                    "問題作成中に"
                    "エラーが発生しました。"
                )

                st.code(
                    str(e)
                )


        # -----------------------------------------
        # 作成された問題を表示
        # -----------------------------------------

        if (
            st.session_state.exam_questions
        ):

            st.markdown(
                st.session_state
                .exam_questions
            )


            student_answer = (
                st.text_area(
                    "あなたの回答",
                    placeholder=(
                        "例："
                        "問1 ③、"
                        "問2 ①、"
                        "問3 ⑤"
                    ),
                )
            )


            if st.button(
                "✅ 採点・解説"
            ):

                if not student_answer:

                    st.warning(
                        "回答を入力してください。"
                    )

                else:

                    explanation_prompt = (
                        build_explanation_prompt(
                            question_text=(
                                st.session_state
                                .exam_questions
                            ),
                            student_answer=(
                                student_answer
                            ),
                            explanation_level=(
                                explanation_level
                            ),
                        )
                    )


                    try:

                        with st.spinner(
                            "採点・解説しています..."
                        ):

                            explanation = (
                                run_agent(
                                    explanation_prompt,
                                    use_conversation=False,
                                )
                            )


                        st.markdown(
                            explanation
                        )


                    except Exception as e:

                        st.error(
                            "採点・解説中に"
                            "エラーが発生しました。"
                        )

                        st.code(
                            str(e)
                        )


    # =====================================================
    # 症例問題
    # =====================================================

    elif mode == "症例問題":

        difficulty = (
            st.selectbox(
                "難易度",
                DIFFICULTY_LEVELS,
            )
        )


        if st.button(
            "🧠 症例問題を作成",
            type="primary",
        ):

            prompt = (
                build_case_question_prompt(
                    field=topic,
                    difficulty=(
                        difficulty
                    ),
                    question_count=3,
                )
            )


            try:

                with st.spinner(
                    "症例問題を作成しています..."
                ):

                    output = (
                        run_agent(
                            prompt,
                            use_conversation=False,
                        )
                    )


                st.markdown(
                    output
                )


            except Exception as e:

                st.error(
                    "症例問題作成中に"
                    "エラーが発生しました。"
                )

                st.code(
                    str(e)
                )


    # =====================================================
    # 苦手分野・間違い復習
    # =====================================================

    elif mode in [
        "苦手分野対策",
        "間違い復習",
    ]:

        weak_content = (
            st.text_area(
                "苦手な内容・間違えた内容",
                height=150,
                placeholder=(
                    "例："
                    "脳卒中の病巣と症状、"
                    "高次脳機能障害の評価"
                ),
            )
        )


        if st.button(
            "🎯 復習プランを作成",
            type="primary",
        ):

            prompt = f"""
作業療法士国家試験対策です。


【学生が苦手または誤答した内容】

{weak_content}


【現在のテーマ】

{topic}


以下を作成してください。

1. 理解が必要な知識

2. 混同しやすいポイント

3. 優先して復習する内容

4. 分かりやすい解説

5. AI作成オリジナル確認問題
"""


            try:

                with st.spinner(
                    "復習内容を作成しています..."
                ):

                    output = (
                        run_agent(
                            prompt,
                            use_conversation=False,
                        )
                    )


                st.markdown(
                    output
                )


            except Exception as e:

                st.error(
                    "復習内容作成中に"
                    "エラーが発生しました。"
                )

                st.code(
                    str(e)
                )


    # =====================================================
    # ミニ模試
    # =====================================================

    elif mode == "ミニ模試":

        question_count = (
            st.selectbox(
                "問題数",
                [
                    10,
                    20,
                ],
            )
        )


        difficulty = (
            st.selectbox(
                "難易度",
                DIFFICULTY_LEVELS,
            )
        )


        if st.button(
            "🏆 ミニ模試を作成",
            type="primary",
        ):

            prompt = (
                build_mock_exam_prompt(
                    question_count=(
                        question_count
                    ),
                    difficulty=(
                        difficulty
                    ),
                )
            )


            try:

                with st.spinner(
                    "ミニ模試を作成しています..."
                ):

                    output = (
                        run_agent(
                            prompt,
                            use_conversation=False,
                        )
                    )


                st.markdown(
                    output
                )


            except Exception as e:

                st.error(
                    "ミニ模試作成中に"
                    "エラーが発生しました。"
                )

                st.code(
                    str(e)
                )


# =========================================================
# Practice MVP
# =========================================================

elif mode == "実践について相談する":

    st.subheader(
        "🩺 実践について相談する"
    )

    st.caption(
        "相談内容を整理し、"
        "次に考えることを明確にします。"
    )

    st.info(
        "個人を特定できる情報は入力せず、"
        "必要最小限の匿名化情報を使用してください。"
    )


    # -----------------------------------------------------
    # 相談内容
    # -----------------------------------------------------

    practice_question = st.text_area(
        "相談内容を入力してください",
        value=(
            st.session_state
            .practice_question
        ),
        height=180,
        placeholder=(
            "例：退院後の生活に不安がある方について、"
            "どのような情報を整理して支援方針を"
            "考えるとよいでしょうか。"
        ),
        key="practice_question_input",
    )


    # -----------------------------------------------------
    # Scope
    # -----------------------------------------------------

    practice_scope = st.selectbox(
        "対象",
        PRACTICE_SCOPE_OPTIONS,
        index=0,
        key="practice_scope_selector",
    )


    # -----------------------------------------------------
    # Context
    # -----------------------------------------------------

    practice_contexts = st.multiselect(
        "関連する場面",
        PRACTICE_CONTEXT_OPTIONS,
        default=[],
        key="practice_context_selector",
        help=(
            "複数の場面が関係する場合は、"
            "複数選択できます。"
        ),
    )


    # -----------------------------------------------------
    # AIで整理する
    # -----------------------------------------------------

    organize_practice = st.button(
        "🧠 AIで整理する",
        type="primary",
        use_container_width=True,
        disabled=(
            not privacy_confirmed
        ),
        key="practice_organize_button",
    )


    if not privacy_confirmed:

        st.warning(
            "サイドバーの"
            "「個人を特定できる情報を入力しません」"
            "にチェックすると利用できます。"
        )


    # -----------------------------------------------------
    # 入力確認
    # -----------------------------------------------------

    if organize_practice:

        clean_question = (
            practice_question.strip()
        )

        if not clean_question:

            st.warning(
                "相談内容を入力してください。"
            )

        elif has_sensitive_information(
            clean_question
        ):

            warning = (
                get_privacy_warning(
                    clean_question
                )
            )

            st.error(
                warning
            )

        else:

        

            try:

                practice_prompt = (
                    build_practice_prompt(
                        question=clean_question,
                        scope=practice_scope,
                        contexts=practice_contexts,
                    )
                )

                with st.spinner(
                    "EGAI-OT Practiceが"
                    "相談内容を整理しています..."
                ):

                    practice_raw_answer = (
                        run_agent(
                            practice_prompt,
                            use_conversation=False,
                        )
                    )

                practice_result = (
                    parse_practice_response(
                        practice_raw_answer
                    )
                )

                st.session_state.practice_question = (
                    clean_question
                )

                st.session_state.practice_result = (
                    practice_result
                )

                st.session_state.practice_evidence_result = None

                st.session_state.practice_history = []
                st.success(
                    "AIによる整理が完了しました。"
                )

            except Exception as e:

                st.error(
                    "相談内容の整理中に"
                    "エラーが発生しました。"
                )

                st.code(
                    str(e)
                )


    # -----------------------------------------------------
    # Practice結果表示
    # -----------------------------------------------------

    if st.session_state.practice_result:

        result = (
            st.session_state.practice_result
        )

        st.divider()

        st.markdown(
            "## Practice整理結果"
        )


        # -------------------------------------------------
        # Scope / Context
        # -------------------------------------------------

        scope_col, context_col = st.columns(2)

        with scope_col:

            st.markdown(
                "#### 対象"
            )

            st.write(
                result.get(
                    "scope",
                    "未指定",
                )
            )

        with context_col:

            st.markdown(
                "#### 関連する場面"
            )

            contexts = result.get(
                "contexts",
                [],
            )

            if contexts:

                st.write(
                    " / ".join(
                        contexts
                    )
                )

            else:

                st.write(
                    "未指定"
                )


        # -------------------------------------------------
        # 今、大切なこと
        # -------------------------------------------------

        st.markdown(
            "### 🎯 今、大切なこと"
        )

        important_now = result.get(
            "important_now",
            "",
        )

        if important_now:

            st.info(
                important_now
            )

        else:

            st.write(
                "現時点では整理できていません。"
            )


        # -------------------------------------------------
        # 今分かっていること
        # -------------------------------------------------

        st.markdown(
            "### ✅ 今分かっていること"
        )

        known_facts = result.get(
            "known_facts",
            [],
        )

        if known_facts:

            for item in known_facts:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "入力情報から確認できる事実が"
                "まだ十分ではありません。"
            )


        # -------------------------------------------------
        # 整理が必要なこと
        # -------------------------------------------------

        st.markdown(
            "### ⚠️ 整理が必要なこと"
        )

        needs_organization = result.get(
            "needs_organization",
            [],
        )

        if needs_organization:

            for item in needs_organization:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "現時点で大きな整理課題は"
                "示されていません。"
            )


        # -------------------------------------------------
        # Decision Focus
        # -------------------------------------------------

        st.markdown(
            "### 🧭 今回考えること"
        )

        decision_focus = result.get(
            "decision_focus",
            "",
        )

        if decision_focus:

            st.success(
                decision_focus
            )

        else:

            st.write(
                "検討テーマをさらに"
                "整理する必要があります。"
            )


        # -------------------------------------------------
        # 次に確認したいこと
        # -------------------------------------------------

        st.markdown(
            "### ❓ 次に確認したいこと"
        )

        missing_information = result.get(
            "missing_information",
            [],
        )

        if missing_information:

            for number, item in enumerate(
                missing_information,
                start=1,
            ):

                st.markdown(
                    f"**{number}.** {item}"
                )

        else:

            st.write(
                "追加で確認すべき重要情報は"
                "現時点では示されていません。"
            )


        # -------------------------------------------------
        # 選択肢
        # -------------------------------------------------

        st.markdown(
            "### 💡 考えられる選択肢"
        )

        options = result.get(
            "options",
            [],
        )

        if options:

            for item in options:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "現時点では具体的な選択肢を"
                "示すための情報が十分ではありません。"
            )


        # -------------------------------------------------
        # 次にすること
        # -------------------------------------------------

        st.markdown(
            "### ➡️ 次にすること"
        )

        next_actions = result.get(
            "next_actions",
            [],
        )

        if next_actions:

            for number, item in enumerate(
                next_actions,
                start=1,
            ):

                st.markdown(
                    f"**{number}.** {item}"
                )

        else:

            st.write(
                "次の行動を決めるには、"
                "追加情報の確認が必要です。"
            )

        # -------------------------------------------------
        # Practice 履歴
        # -------------------------------------------------

        if st.session_state.practice_history:

            st.divider()

            st.markdown(
                "### 📜 これまでの整理"
            )

            st.caption(
                "追加情報を反映する前の整理結果を"
                "振り返ることができます。"
            )

            for history_number, history_entry in enumerate(
                st.session_state.practice_history,
                start=1,
            ):

                history_result = history_entry.get(
                    "result",
                    {},
                )

                history_additional_information = (
                    history_entry.get(
                        "additional_information",
                        "",
                    )
                )

                with st.expander(
                    f"第{history_number}回の整理"
                ):

                    history_focus = history_result.get(
                        "decision_focus",
                        "",
                    )

                    if history_focus:

                        st.markdown(
                            "**今回考えること**"
                        )

                        st.write(
                            history_focus
                        )

                    history_actions = history_result.get(
                        "next_actions",
                        [],
                    )

                    if history_actions:

                        st.markdown(
                            "**次にすること**"
                        )

                        for number, item in enumerate(
                            history_actions,
                            start=1,
                        ):

                            st.markdown(
                                f"{number}. {item}"
                            )

                    if history_additional_information:

                        st.markdown(
                            "**この後に追加した情報**"
                        )

                        st.info(
                            history_additional_information
                        )

        # -------------------------------------------------
        # Evidence
        # -------------------------------------------------

        st.divider()

        st.markdown(
            "### 📚 根拠を確認する"
        )

        evidence_question = (
            result.get(
                "decision_focus",
                "",
            )
            or result.get(
                "important_now",
                "",
            )
            or st.session_state.practice_question
        )

        st.caption(
            "現在の整理結果をもとに、"
            "登録資料とWeb上の根拠を確認します。"
        )

        st.info(
            evidence_question
        )

        check_evidence = st.button(
            "📚 この論点の根拠を確認する",
            use_container_width=True,
            disabled=(
                not privacy_confirmed
            ),
            key="practice_evidence_button",
        )

        if check_evidence:

            try:

                practice_context = {
                    "original_question": (
                        st.session_state.practice_question
                    ),
                    "scope": result.get(
                        "scope",
                        "",
                    ),
                    "contexts": result.get(
                        "contexts",
                        [],
                    ),
                    "known_facts": result.get(
                        "known_facts",
                        [],
                    ),
                    "decision_focus": result.get(
                        "decision_focus",
                        "",
                    ),
                }

                with st.spinner(
                    "根拠を確認しています..."
                ):

                    evidence_result = (
                        run_evidence_search(
                            question=evidence_question,
                            practice_context=json.dumps(
                                practice_context,
                                ensure_ascii=False,
                            ),
                        )
                    )

                st.session_state.practice_evidence_result = (
                    evidence_result
                )

                st.rerun()

            except Exception as e:

                st.error(
                    "根拠の確認中に"
                    "エラーが発生しました。"
                )

                st.code(
                    str(e)
                )

        # -------------------------------------------------
        # Evidence 結果表示
        # -------------------------------------------------

        if st.session_state.practice_evidence_result:

            evidence_result = (
                st.session_state.practice_evidence_result
            )

            st.markdown(
                "#### 📖 Evidence結果"
            )

            tools_used = evidence_result.get(
                "tools_used",
                {},
            )

            web_status = (
                "使用"
                if tools_used.get(
                    "web_search",
                    False,
                )
                else "未使用"
            )

            file_status = (
                "使用"
                if tools_used.get(
                    "file_search",
                    False,
                )
                else "未使用"
            )

            st.caption(
                f"Web Search：{web_status} ／ "
                f"登録資料：{file_status}"
            )

            evidence_answer = evidence_result.get(
                "answer",
                "",
            )

            if evidence_answer:

                st.markdown(
                    evidence_answer
                )

            else:

                st.warning(
                    "根拠に基づく回答を"
                    "取得できませんでした。"
                )


            # ---------------------------------------------
            # Web出典
            # ---------------------------------------------

            web_sources = evidence_result.get(
                "web_sources",
                [],
            )

            if web_sources:

                st.markdown(
                    "#### 🌐 Web出典"
                )

                for number, source in enumerate(
                    web_sources,
                    start=1,
                ):

                    title = (
                        source.get(
                            "title",
                            "",
                        )
                        or f"Web出典 {number}"
                    )

                    url = source.get(
                        "url",
                        "",
                    )

                    if url:

                        st.markdown(
                            f"{number}. [{title}]({url})"
                        )

                    else:

                        st.markdown(
                            f"{number}. {title}"
                        )


            # ---------------------------------------------
            # 登録資料
            # ---------------------------------------------

            file_sources = evidence_result.get(
                "file_sources",
                [],
            )

            if file_sources:

                st.markdown(
                    "#### 📄 登録資料"
                )

                for number, source in enumerate(
                    file_sources,
                    start=1,
                ):

                    filename = (
                        source.get(
                            "filename",
                            "",
                        )
                        or f"登録資料 {number}"
                    )

                    source_text = source.get(
                        "text",
                        "",
                    )

                    with st.expander(
                        f"{number}. {filename}"
                    ):

                        if source_text:

                            display_text = (
                                source_text[:1200]
                            )

                            if len(
                                source_text
                            ) > 1200:

                                display_text += (
                                    "\n\n…"
                                )

                            st.write(
                                display_text
                            )

                        else:

                            st.write(
                                "参照箇所の本文は"
                                "取得されませんでした。"
                            )

        # -------------------------------------------------
        # 追加情報による再整理
        # -------------------------------------------------

        st.divider()

        st.markdown(
            "### 🔄 追加情報をもとに再整理"
        )

        st.caption(
            "「次に確認したいこと」などについて"
            "追加で分かった情報を入力すると、"
            "整理結果を更新できます。"
        )

        additional_information = st.text_area(
            "追加で分かったこと",
            placeholder=(
                "例：入浴時に特に不安があり、"
                "浴槽をまたぐ際にふらつきがあります。"
            ),
            height=140,
            key="practice_additional_information",
        )

        refine_practice = st.button(
            "🔄 AIで再整理する",
            type="primary",
            use_container_width=True,
            disabled=(
                not privacy_confirmed
            ),
            key="practice_refine_button",
        )


        if refine_practice:

            clean_additional_information = (
                additional_information.strip()
            )

            if not clean_additional_information:

                st.warning(
                    "追加情報を入力してください。"
                )

            elif has_sensitive_information(
                clean_additional_information
            ):

                warning = (
                    get_privacy_warning(
                        clean_additional_information
                    )
                )

                st.error(
                    warning
                )

            else:

                try:

                    refinement_prompt = (
                        build_practice_refinement_prompt(
                            original_question=(
                                st.session_state
                                .practice_question
                            ),
                            current_result=result,
                            additional_information=(
                                clean_additional_information
                            ),
                        )
                    )

                    with st.spinner(
                        "追加情報をもとに"
                        "再整理しています..."
                    ):

                        refinement_raw_answer = (
                            run_agent(
                                refinement_prompt,
                                use_conversation=False,
                            )
                        )

                    refined_result = (
                        parse_practice_response(
                            refinement_raw_answer
                        )
                    )

                    # Scopeは初回設定を維持
                    refined_result["scope"] = (
                        result.get(
                            "scope",
                            "個人",
                        )
                    )

                    # Contextが空になった場合は
                    # 現在のContextを維持
                    if not refined_result.get(
                        "contexts"
                    ):

                        refined_result["contexts"] = (
                            result.get(
                                "contexts",
                                [],
                            )
                        )

                    history_entry = {
                        "question": (
                            st.session_state.practice_question
                        ),
                        "result": json.loads(
                            json.dumps(
                                result,
                                ensure_ascii=False,
                            )
                        ),
                        "additional_information": (
                            clean_additional_information
                        ),
                    }

                    st.session_state.practice_history.append(
                        history_entry
                    )

                    st.session_state.practice_result = (
                        refined_result
                    )

                    st.session_state.practice_evidence_result = None

                    st.rerun()

                except Exception as e:

                    st.error(
                        "再整理中に"
                        "エラーが発生しました。"
                    )

                    st.code(
                        str(e)
                    )

# =========================================================
# Research Runner
# =========================================================

elif mode == "🔬 Research Runner":

    st.subheader(
        "🔬 Research Runner"
    )

    st.caption(
        "研究テーマから研究疑問を整理し、"
        "先行研究を検索したうえで、"
        "研究ギャップと次の研究計画を整理します。"
    )

    research_topic = st.text_area(
        "研究テーマ",
        placeholder=(
            "例：高次脳機能障害者の"
            "テレワーク就労支援"
        ),
        key="research_runner_topic",
    )

    run_research_clicked = st.button(
        "研究を実行",
        type="primary",
        key="research_runner_run",
    )

    if run_research_clicked:

        clean_research_topic = (
            research_topic.strip()
        )

        if not clean_research_topic:

            st.warning(
                "研究テーマを入力してください。"
            )

        else:

            try:

                with st.spinner(
                    "研究疑問の整理、"
                    "先行研究の検索、"
                    "研究計画の整理を実行しています。"
                ):

                    research_result = (
                        run_research_runner(
                            clean_research_topic
                        )
                    )

                st.session_state.research_runner_result = (
                    research_result
                )

            except Exception as e:

                st.error(
                    "Research Runnerの実行中に"
                    "エラーが発生しました。"
                )

                st.code(
                    str(e)
                )

    if st.session_state.research_runner_result:

        research_result = (
            st.session_state.research_runner_result
        )

        st.divider()

        question_tab, search_tab, synthesis_tab = (
            st.tabs(
                [
                    "① 研究疑問",
                    "② 先行研究",
                    "③ 研究計画",
                ]
            )
        )

        with question_tab:

            st.markdown(
                research_result.get(
                    "question",
                    ""
                )
            )

        with search_tab:

            search_result = (
                research_result.get(
                    "search",
                    {}
                )
            )

            st.markdown(
                search_result.get(
                    "answer",
                    ""
                )
            )

        with synthesis_tab:

            st.markdown(
                research_result.get(
                    "synthesis",
                    ""
                )
            )

# =========================================================
# 通常AIチャット
# =========================================================

else:

    # -----------------------------------------------------
    # モード説明
    # -----------------------------------------------------

    mode_instruction = (
        get_mode_instruction(
            mode
        )
    )


    # -----------------------------------------------------
    # 会話履歴表示
    # -----------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message[
                "role"
            ]
        ):

            st.markdown(
                message[
                    "content"
                ]
            )


    # -----------------------------------------------------
    # チャット入力
    # -----------------------------------------------------

    question = st.chat_input(
        "相談内容を入力してください",
        disabled=(
            not privacy_confirmed
        ),
    )


    # -----------------------------------------------------
    # 未チェック時の案内
    # -----------------------------------------------------

    if not privacy_confirmed:

        st.info(
            "サイドバーの"
            "「個人を特定できる情報を入力しません」"
            "にチェックすると、"
            "相談内容を入力できます。"
        )


    # -----------------------------------------------------
    # 質問が入力された場合
    # -----------------------------------------------------

    if question:

        # =============================================
        # 個人情報チェック
        # =============================================

        if has_sensitive_information(
            question
        ):

            warning = (
                get_privacy_warning(
                    question
                )
            )

            st.error(
                warning
            )


        else:

            # =========================================
            # ユーザー発言
            # =========================================

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )


            with st.chat_message(
                "user"
            ):

                st.markdown(
                    question
                )


            # =========================================
            # 基本質問
            # =========================================

            basic_question = (
                build_question(
                    category,
                    mode,
                    question,
                )
            )


            # =========================================
            # モード別専門指示
            # =========================================

            full_prompt = f"""
{basic_question}


【このモード専用の指示】

{mode_instruction}


【回答上の重要事項】

・入力されていない事実を作らないでください。

・確認できないことは、
  「確認できない」
  「追加情報が必要」
  と明示してください。

・登録資料に基づく情報、
  Web検索で確認した情報、
  AIによる推論・仮説を
  必要に応じて区別してください。

・専門職による最終判断を
  支援する回答としてください。
"""


            # =========================================
            # AI回答
            # =========================================

            with st.chat_message(
                "assistant"
            ):

                try:

                    with st.spinner(
                        "OT Practice Copilotが"
                        "検討しています..."
                    ):

                        answer = (
                            run_agent(
                                full_prompt,
                                use_conversation=True,
                            )
                        )


                    st.markdown(
                        answer
                    )


                    # ---------------------------------
                    # 会話履歴保存
                    # ---------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )


                except Exception as e:

                    st.error(
                        "回答生成中に"
                        "エラーが発生しました。"
                    )

                    st.code(
                        str(e)
                    )


# =========================================================
# フッター
# =========================================================

st.divider()


st.caption(
    "OT Practice Copilotは、"
    "作業療法士・研究者・教員・学生の"
    "専門的な思考と意思決定を支援するツールです。"
)


st.caption(
    "AIの回答のみで、"
    "診断、治療方針、復職可否、"
    "研究上・教育上の重要な判断などを"
    "最終決定しないでください。"
)