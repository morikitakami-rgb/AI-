from dotenv import load_dotenv

from agents import (
    Agent,
    FileSearchTool,
    Runner,
    WebSearchTool,
)

from evidence_core import (
    VECTOR_STORE_ID,
    check_evidence_environment,
    extract_evidence_result,
)

load_dotenv()

# =========================================================
# Research Runner Core
# =========================================================


def build_research_question_prompt(topic):
    """
    研究テーマを研究可能な問いへ整理するための
    Research Runner用プロンプトを作成する。

    この段階ではWeb Search / File Searchは行わない。
    """

    clean_topic = str(topic).strip()

    if not clean_topic:
        raise ValueError(
            "研究テーマが入力されていません。"
        )

    return f"""
あなたは「EGAI-OT Research Runner」です。

以下の研究テーマについて、
現時点で与えられている情報だけを用いて、
研究可能な問いへ整理してください。

この段階では外部検索を行わず、
入力されていない事実を推測で補わないでください。

【研究テーマ】
{clean_topic}

以下の項目で整理してください。

1. 研究テーマ
2. 背景として確認する必要があること
3. 研究上の問題
4. 研究疑問の候補
5. 適しそうな研究疑問の形式
   （PICO、PECO、PEO、SPIDER等。無理に当てはめない）
6. 検索に使用するとよいキーワード
7. 現時点で分からないこと
8. 次の文献検索で確認すべき問い

重要：
・まだ先行研究を確認していないため、
  新規性や独自性があると断定しないでください。
・確認できていない内容は
  「未確認」と明示してください。
・個人を特定できる情報は求めないでください。
""".strip()

# =========================================================
# Research Search Prompt
# =========================================================


def build_research_search_prompt(topic, question_summary=None):
    """
    研究テーマについて、
    先行研究を検索・整理するためのプロンプトを作成する。
    """

    clean_topic = str(topic).strip()

    if not clean_topic:
        raise ValueError(
            "研究テーマが入力されていません。"
        )

    summary_text = ""

    if question_summary:
        summary_text = (
            "\n\n【事前に整理した研究疑問】\n"
            f"{str(question_summary).strip()}"
        )

    return f"""
あなたは「EGAI-OT Research Runner」です。

以下の研究テーマについて、
先行研究を確認するための検索と整理を行ってください。

【研究テーマ】
{clean_topic}
{summary_text}

以下の観点で確認してください。

1. 主要な先行研究
2. 現在までに分かっていること
3. 研究結果が一致している点
4. 研究結果が一致していない点
5. まだ十分に分かっていないこと
6. 研究対象・方法・評価指標の傾向
7. 今後さらに確認すべき検索語
8. 研究ギャップ候補

重要：
・検索で確認できた情報と、
  AIによる解釈を区別してください。
・新規性や独自性は、
  十分な先行研究確認なしに断定しないでください。
・研究ギャップは
  「候補」として示してください。
・一次資料、学術論文、公的機関、
  専門学会など信頼性の高い情報を優先してください。
・個人を特定できる情報は求めないでください。
""".strip()

# =========================================================
# Research Synthesis Prompt
# =========================================================


def build_research_synthesis_prompt(
    topic,
    search_summary,
    question_summary=None,
):
    """
    先行研究の検索結果をもとに、
    研究ギャップ候補と次の研究計画を整理する
    プロンプトを作成する。
    """

    clean_topic = str(topic).strip()
    clean_search_summary = str(search_summary).strip()

    if not clean_topic:
        raise ValueError(
            "研究テーマが入力されていません。"
        )

    if not clean_search_summary:
        raise ValueError(
            "先行研究の検索結果が入力されていません。"
        )

    question_text = ""

    if question_summary:
        question_text = (
            "\n\n【事前に整理した研究疑問】\n"
            f"{str(question_summary).strip()}"
        )

    return f"""
あなたは「EGAI-OT Research Runner」です。

以下の研究テーマと先行研究の検索結果をもとに、
研究ギャップ候補と次の研究計画を整理してください。

【研究テーマ】
{clean_topic}
{question_text}

【先行研究の検索結果】
{clean_search_summary}

以下の項目で整理してください。

1. 先行研究から確認できたこと
2. まだ十分に分かっていないこと
3. 研究ギャップ候補
4. 新規性・独自性の候補
5. 研究疑問の候補
6. 適しそうな研究デザイン候補
7. 主な対象・評価指標・解析の候補
8. 実施可能性を確認すべき点
9. 倫理面で確認すべき点
10. 次に行うべき具体的な研究作業

重要：
・新規性や独自性は断定せず、
  「候補」として示してください。
・検索結果で確認できた事実と、
  AIによる解釈を区別してください。
・検索結果に含まれていない内容を
  事実として補わないでください。
・不明な点は「未確認」と明示してください。
・研究デザインは目的に応じて複数候補を示し、
  1つに決めつけないでください。
・個人を特定できる情報は求めないでください。
""".strip()

# =========================================================
# Research Question Agent
# =========================================================


RESEARCH_QUESTION_INSTRUCTIONS = """
あなたは「EGAI-OT Research Runner」です。

研究テーマを、研究可能な問いへ整理することを支援します。

この段階では外部検索を行わず、
入力された情報だけを使用してください。

入力されていない事実を補わず、
不明な点は未確認として扱ってください。

新規性や独自性を断定せず、
個人を特定できる情報を求めないでください。
""".strip()


def create_research_question_agent():
    """
    研究疑問整理専用Agentを作成する。
    """

    return Agent(
        name="EGAI-OT Research Question",
        instructions=RESEARCH_QUESTION_INSTRUCTIONS,
    )

# =========================================================
# Research Question Run
# =========================================================


def run_research_question(topic):
    """
    研究テーマをAIで整理し、
    研究疑問整理の結果を返す。
    """

    agent = create_research_question_agent()

    prompt = build_research_question_prompt(
        topic
    )

    result = Runner.run_sync(
        agent,
        prompt,
    )

    return str(
        result.final_output
        or ""
    )

# =========================================================
# Research Search Agent
# =========================================================


RESEARCH_SEARCH_INSTRUCTIONS = """
あなたは「EGAI-OT Research Runner」の文献検索担当です。

研究テーマと事前に整理された研究疑問をもとに、
先行研究を検索し、研究の現状を整理してください。

【基本方針】

1. Web Search と File Search を使用して根拠を確認する。
2. Web検索では、一次資料、学術論文、公的機関、
   専門学会など信頼性の高い情報を優先する。
3. 登録資料に関連情報があれば、その内容も確認する。
4. 検索で確認できた事実とAIによる解釈を区別する。
5. 確認できない内容を推測で補わない。
6. 新規性・独自性・研究ギャップは断定せず、
   候補として示す。
7. 不明な点は「未確認」と明示する。
8. 個人を特定できる情報を求めない。
""".strip()


def create_research_search_agent():
    """
    先行研究検索専用Agentを作成する。
    """

    check_evidence_environment()

    return Agent(
        name="EGAI-OT Research Search",
        instructions=RESEARCH_SEARCH_INSTRUCTIONS,
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
# Research Search Run
# =========================================================


def run_research_search(
    topic,
    question_summary=None,
):
    """
    先行研究検索を実行し、
    回答と出典情報を返す。
    """

    agent = create_research_search_agent()

    prompt = build_research_search_prompt(
        topic=topic,
        question_summary=question_summary,
    )

    result = Runner.run_sync(
        agent,
        prompt,
    )

    return extract_evidence_result(
        result
    )

# =========================================================
# Research Synthesis Agent
# =========================================================


RESEARCH_SYNTHESIS_INSTRUCTIONS = """
あなたは「EGAI-OT Research Runner」の研究統合担当です。

研究テーマ、事前に整理された研究疑問、
先行研究の検索結果をもとに、
研究ギャップ候補と次の研究計画を整理してください。

【基本方針】

1. 検索結果で確認できた事実とAIによる解釈を区別する。
2. 検索結果に含まれていない内容を事実として補わない。
3. 新規性・独自性・研究ギャップは断定せず、
   候補として示す。
4. 不明な点は「未確認」と明示する。
5. 研究デザインは目的に応じて複数候補を示し、
   1つに決めつけない。
6. 実施可能性と倫理面の確認事項も示す。
7. 個人を特定できる情報を求めない。
""".strip()


def create_research_synthesis_agent():
    """
    研究統合専用Agentを作成する。
    """

    return Agent(
        name="EGAI-OT Research Synthesis",
        instructions=RESEARCH_SYNTHESIS_INSTRUCTIONS,
    )