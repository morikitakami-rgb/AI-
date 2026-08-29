import os

from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
    FileSearchTool,
)


# ========================================
# .env の読み込み
# ========================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")


# ========================================
# 環境変数の確認
# ========================================

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEYが.envに設定されていません。"
    )

if not VECTOR_STORE_ID:
    raise ValueError(
        "VECTOR_STORE_IDが.envに設定されていません。"
    )


# ========================================
# AIエージェントの設定
# ========================================

agent = Agent(
    name="OT Research Agent",

    instructions="""
あなたは、作業療法・リハビリテーション・研究・教育を
支援するAIエージェントです。

【基本方針】

・質問に対して、分かりやすい日本語で回答してください。

・登録されている研究資料、論文、研究計画書について
  質問された場合は、File Searchを使用して内容を確認してください。

・最新情報、先行研究、制度、製品、ニュースなど、
  インターネットで確認する必要がある場合は、
  Web Searchを使用してください。

・必要に応じて、
  File SearchとWeb Searchの両方を使用してください。

・登録資料に書かれていない内容を、
  書かれているかのように回答しないでください。

・確認できないことは推測で断定せず、
  「確認できません」「分かりません」と明示してください。

・研究に関する質問では、
  必要に応じて以下の観点から検討してください。

  - 研究目的
  - 研究対象
  - 研究デザイン
  - 評価指標
  - 統計解析
  - 倫理的配慮
  - 新規性
  - 独自性
  - 臨床的意義
  - 研究の限界

・Web検索を使用した場合は、
  可能な限り情報源を示してください。

・研究や医療に関する情報では、
  査読付き論文、公的機関、学術団体、
  公式ガイドラインなど、
  信頼性の高い情報源を優先してください。

・File Searchの資料とWeb検索の情報が異なる場合は、
  その違いが分かるように説明してください。

・最終的な臨床判断や研究判断は、
  専門職・研究者が行うことを前提として支援してください。
""",

    tools=[
        WebSearchTool(
            search_context_size="medium"
        ),

        FileSearchTool(
            vector_store_ids=[VECTOR_STORE_ID],
            max_num_results=5,
        ),
    ],
)


# ========================================
# 会話履歴
# ========================================

session = SQLiteSession(
    "my_conversation",
    "conversation.db"
)


# ========================================
# 起動画面
# ========================================

print("========================================")
print(" OT Research Agent")
print("========================================")
print(" Web検索：ON")
print(" PDF・資料検索：ON")
print(" 会話履歴保存：ON")
print("========================================")
print("終了するときは「終了」と入力してください。")
print()


# ========================================
# メイン処理
# ========================================

while True:

    question = input("あなた：")

    # -------------------------------
    # 終了
    # -------------------------------

    if question.strip().lower() in [
        "終了",
        "exit",
        "quit",
    ]:
        print()
        print("AI：終了します。")
        break

    # -------------------------------
    # 空欄
    # -------------------------------

    if not question.strip():
        continue

    # -------------------------------
    # AIへ質問
    # -------------------------------

    try:

        result = Runner.run_sync(
            agent,
            question,
            session=session,
        )

        print()
        print("AI：")
        print(result.final_output)
        print()

    # -------------------------------
    # Ctrl + C
    # -------------------------------

    except KeyboardInterrupt:

        print()
        print("処理を中断しました。")
        print()

    # -------------------------------
    # その他のエラー
    # -------------------------------

    except Exception as e:

        print()
        print("エラーが発生しました。")
        print(e)
        print()