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
    analyze_video,
    summarize_by_phase,
    get_video_info,
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


if "exam_questions" not in st.session_state:

    st.session_state.exam_questions = None


if "pptx_bytes" not in st.session_state:

    st.session_state.pptx_bytes = None


if "pptx_filename" not in st.session_state:

    st.session_state.pptx_filename = None


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
        "ローカル版では、元動画そのものを"
        "OpenAIへ送信せず、"
        "MediaPipeでPC内解析します。"
        "AIへ送信するのは解析後の数値のみです。"
    )


    # -----------------------------------------------------
    # 動画確認
    # -----------------------------------------------------

    video_confirmed = (
        st.checkbox(
            "テスト用または適切に匿名化された"
            "動画であることを確認しました"
        )
    )


    uploaded_video = (
        st.file_uploader(
            "解析する動画",
            type=[
                "mp4",
                "mov",
                "avi",
                "mkv",
            ],
        )
    )


    # -----------------------------------------------------
    # 動画が選択された場合
    # -----------------------------------------------------

    if uploaded_video:

        st.video(
            uploaded_video
        )


        suffix = Path(
            uploaded_video.name
        ).suffix


        # -----------------------------------------
        # 動画情報確認用の一時ファイル
        # -----------------------------------------

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


        try:

            video_info = (
                get_video_info(
                    inspect_path
                )
            )

        finally:

            if os.path.exists(
                inspect_path
            ):

                os.remove(
                    inspect_path
                )


        # -----------------------------------------
        # 動画基本情報
        # -----------------------------------------

        c1, c2, c3 = (
            st.columns(3)
        )


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
            video_info[
                "frame_count"
            ],
        )


        # -----------------------------------------
        # 解析頻度
        # -----------------------------------------

        analysis_fps = (
            st.selectbox(
                "解析頻度",
                [
                    10,
                    15,
                    30,
                ],
                index=2,
            )
        )


        # =========================================
        # 工程設定
        # =========================================

        st.subheader(
            "工程設定"
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

        if st.button(
            "📊 動作解析を開始",
            type="primary",
            disabled=(
                not video_confirmed
                or not phases_valid
            ),
        ):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:

                tmp.write(
                    uploaded_video.getvalue()
                )

                video_path = (
                    tmp.name
                )


            try:

                with st.spinner(
                    "MediaPipeで"
                    "全身・手指を解析しています..."
                ):

                    (
                        motion_df,
                        _
                    ) = analyze_video(
                        video_path=video_path,
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


                    st.session_state.motion_df = (
                        motion_df
                    )

                    st.session_state.motion_summary = (
                        summary_df
                    )


                st.success(
                    "動作解析が完了しました。"
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

                if os.path.exists(
                    video_path
                ):

                    os.remove(
                        video_path
                    )


    # =====================================================
    # 動作解析結果
    # =====================================================

    if (
        st.session_state.motion_df
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


        st.subheader(
            "📊 工程別解析結果"
        )


        st.dataframe(
            summary_df,
            use_container_width=True,
        )


        # -----------------------------------------
        # 左右選択
        # -----------------------------------------

        side = st.radio(
            "グラフ表示側",
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


        # -----------------------------------------
        # グラフ
        # -----------------------------------------

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
                "📈 時間―角度変化"
            )


            st.line_chart(
                graph_df
            )


        # -----------------------------------------
        # CSV
        # -----------------------------------------

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
        )


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
        )


        # -----------------------------------------
        # AIによるOT視点の整理
        # -----------------------------------------

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
2D投影値です。

正式なROM測定値や
三次元動作解析値として
扱わないでください。

観測された事実と
AIによる仮説を
明確に区別してください。
"""


            try:

                with st.spinner(
                    "解析結果を"
                    "OTの視点から整理しています..."
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