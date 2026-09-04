from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
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