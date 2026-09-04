import os

from dotenv import load_dotenv

from agents import (
    Agent,
    FileSearchTool,
    Runner,
    WebSearchTool,
)


# =========================================================
# 環境変数
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

VECTOR_STORE_ID = os.getenv(
    "VECTOR_STORE_ID"
)


# =========================================================
# Evidence 共通指示
# =========================================================

EVIDENCE_INSTRUCTIONS = """
あなたは EGAI-OT Evidence です。

作業療法・リハビリテーション・就労支援などの
実践上の疑問について、根拠を確認する支援をします。

【基本方針】

1. 必要に応じて Web Search と File Search を使用する。
2. 登録資料に関連情報がある場合は、その内容を確認する。
3. Web検索では、可能な限り一次資料、学術論文、
   公的機関、専門学会など信頼性の高い情報を優先する。
4. 根拠が確認できない内容を推測で補わない。
5. 事実、解釈、実践への示唆を混同しない。
6. 研究結果を個別症例へそのまま適用できるとは限らないため、
   適用可能性や限界も示す。
7. 診断、治療、復職可否などの最終判断は行わない。
8. 個人を特定できる情報を要求しない。

【回答の基本構成】

■ 根拠から分かること
■ 実践への示唆
■ 注意点・限界

簡潔で、実践者が確認しやすい形で回答してください。
Web情報を使用した場合は、利用した出典が分かるようにしてください。
"""


# =========================================================
# 環境確認
# =========================================================

def check_evidence_environment():
    """
    Evidence機能に必要な環境変数を確認する。
    """

    if not OPENAI_API_KEY:

        raise ValueError(
            "OPENAI_API_KEYが"
            ".envに設定されていません。"
        )

    if not VECTOR_STORE_ID:

        raise ValueError(
            "VECTOR_STORE_IDが"
            ".envに設定されていません。"
        )


# =========================================================
# Evidence Agent
# =========================================================

def create_evidence_agent():
    """
    Web Search と File Search を利用する
    Evidence専用Agentを作成する。
    """

    check_evidence_environment()

    return Agent(
        name="EGAI-OT Evidence",
        instructions=EVIDENCE_INSTRUCTIONS,
        tools=[
            WebSearchTool(
                search_context_size="medium"
            ),
            FileSearchTool(
                vector_store_ids=[
                    VECTOR_STORE_ID
                ],
                max_num_results=5,
                include_search_results=True,
            ),
        ],
    )


# =========================================================
# Evidence Prompt
# =========================================================

def build_evidence_prompt(
    question,
    practice_context=None,
):
    """
    Evidence検索用プロンプトを作成する。
    """

    clean_question = str(
        question
    ).strip()

    if not clean_question:

        raise ValueError(
            "Evidenceを確認する疑問が"
            "入力されていません。"
        )

    context_text = ""

    if practice_context:

        context_text = (
            "\n\n"
            "【現在の実践上の文脈】\n"
            f"{practice_context}"
        )

    return (
        "以下の実践上の疑問について、"
        "根拠を確認してください。\n\n"
        "【疑問】\n"
        f"{clean_question}"
        f"{context_text}\n\n"
        "登録資料とWeb上の情報を必要に応じて検索し、"
        "確認できた根拠だけを用いて整理してください。"
    )


# =========================================================
# 内部ユーティリティ
# =========================================================

def _get_value(
    obj,
    name,
    default=None,
):
    """
    dict / Pydantic model の両方から
    値を取得する。
    """

    if isinstance(
        obj,
        dict,
    ):

        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


# =========================================================
# Evidence 出典抽出
# =========================================================

def extract_evidence_result(
    run_result,
):
    """
    Agents SDK の RunResult から
    回答、Web出典、File Search結果を抽出する。
    """

    web_sources = []

    file_sources = []

    web_seen = set()

    file_seen = set()

    web_search_used = False

    file_search_used = False


    for response in (
        run_result.raw_responses
    ):

        outputs = getattr(
            response,
            "output",
            [],
        )

        for item in outputs:

            item_type = _get_value(
                item,
                "type",
                "",
            )


            # ---------------------------------------------
            # Web Search
            # ---------------------------------------------

            if item_type == "web_search_call":

                web_search_used = True


            # ---------------------------------------------
            # File Search
            # ---------------------------------------------

            elif item_type == "file_search_call":

                file_search_used = True

                results = _get_value(
                    item,
                    "results",
                    [],
                ) or []

                for result in results:

                    file_id = _get_value(
                        result,
                        "file_id",
                        "",
                    )

                    filename = _get_value(
                        result,
                        "filename",
                        "",
                    )

                    score = _get_value(
                        result,
                        "score",
                        None,
                    )

                    text = _get_value(
                        result,
                        "text",
                        "",
                    )

                    key = (
                        file_id,
                        filename,
                        text,
                    )

                    if key in file_seen:

                        continue

                    file_seen.add(
                        key
                    )

                    file_sources.append(
                        {
                            "file_id": file_id,
                            "filename": filename,
                            "score": score,
                            "text": text,
                        }
                    )


            # ---------------------------------------------
            # Message内のWeb引用
            # ---------------------------------------------

            elif item_type == "message":

                contents = _get_value(
                    item,
                    "content",
                    [],
                ) or []

                for content in contents:

                    annotations = _get_value(
                        content,
                        "annotations",
                        [],
                    ) or []

                    for annotation in annotations:

                        annotation_type = (
                            _get_value(
                                annotation,
                                "type",
                                "",
                            )
                        )

                        if (
                            annotation_type
                            != "url_citation"
                        ):

                            continue

                        title = _get_value(
                            annotation,
                            "title",
                            "",
                        )

                        url = _get_value(
                            annotation,
                            "url",
                            "",
                        )

                        if not url:

                            continue

                        if url in web_seen:

                            continue

                        web_seen.add(
                            url
                        )

                        web_sources.append(
                            {
                                "title": title,
                                "url": url,
                            }
                        )


    return {
        "answer": str(
            run_result.final_output
            or ""
        ),
        "web_sources": web_sources,
        "file_sources": file_sources,
        "tools_used": {
            "web_search": (
                web_search_used
            ),
            "file_search": (
                file_search_used
            ),
        },
    }


# =========================================================
# Evidence 実行
# =========================================================

def run_evidence_search(
    question,
    practice_context=None,
):
    """
    Evidence検索を実行し、
    回答と出典情報を返す。
    """

    agent = create_evidence_agent()

    prompt = build_evidence_prompt(
        question=question,
        practice_context=(
            practice_context
        ),
    )

    result = Runner.run_sync(
        agent,
        prompt,
    )

    return extract_evidence_result(
        result
    )