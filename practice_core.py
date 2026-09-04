import json


# =========================================================
# Practice MVP 基本設定
# =========================================================

PRACTICE_SCOPE_OPTIONS = [
    "個人",
    "集団",
    "組織",
]


PRACTICE_CONTEXT_OPTIONS = [
    "医療・リハ",
    "自宅",
    "就労",
    "学校・教育",
    "地域生活",
    "その他",
]


# =========================================================
# Practice用プロンプト
# =========================================================

def build_practice_prompt(
    question,
    scope="個人",
    contexts=None,
):
    """
    Practice MVP用の構造化プロンプトを作成する。

    この段階では、
    利用者の相談内容を整理することを目的とし、
    診断・治療・復職可否などの最終判断は行わない。
    """

    question = (question or "").strip()

    if not question:
        raise ValueError(
            "相談内容が入力されていません。"
        )

    if scope not in PRACTICE_SCOPE_OPTIONS:
        scope = "個人"

    if contexts is None:
        contexts = []

    contexts = [
        str(context).strip()
        for context in contexts
        if str(context).strip()
    ]

    context_text = (
        "、".join(contexts)
        if contexts
        else "未指定"
    )

    return f"""
あなたはEGAI-OTのPractice支援機能です。

以下の相談について、
まず「答えを出す」のではなく、
専門職が考えるための情報整理を行ってください。


【相談内容】

{question}


【現在のScope】

{scope}


【関連するContext】

{context_text}


【この処理の目的】

利用者が、

・何が分かっているか
・何がまだ分からないか
・今回何を考える必要があるか
・次に何を確認するとよいか
・どのような選択肢が考えられるか

を整理できるようにしてください。


【重要な原則】

1.
入力されていない事実を
勝手に補わないでください。

2.
推論や仮説は、
事実とは区別してください。

3.
不足情報は、
意思決定に影響する重要なものだけを
最大3個に絞ってください。

4.
不足情報を無理に3個作る必要はありません。

5.
診断、治療方針、復職可否などを
AIだけで最終判断しないでください。

6.
本人の問題だけに限定せず、
必要に応じて、

・本人
・作業
・環境
・時間

の関係から整理してください。

就労の場合は必要に応じて、

・組織

も考慮してください。

7.
氏名、住所、正確な生年月日、
電話番号、メールアドレス、
患者ID、職員番号、
具体的な勤務先名・施設名などの
個人を直接特定できる情報を
追加で求めないでください。

8.
この段階では、
Web検索や登録資料の検索を
積極的には行わず、
まず利用者が入力した情報の整理を
優先してください。


【出力形式】

必ず以下のJSONオブジェクトのみを
出力してください。

Markdownのコードブロックは不要です。

{{
  "scope": "{scope}",
  "contexts": [],
  "important_now": "",
  "known_facts": [],
  "needs_organization": [],
  "decision_focus": "",
  "missing_information": [],
  "options": [],
  "next_actions": []
}}


【各項目の意味】

scope:
現在の検討単位。

contexts:
今回関係する場面。
入力されたContextを基本としてください。

important_now:
この相談で現時点で最も重要なことを、
短い文章で示してください。

known_facts:
利用者の入力から確認できる事実だけを
箇条書きで示してください。

needs_organization:
まだ整理が必要な点、
不確実な点、
仮説として検討すべき点を示してください。

decision_focus:
今回まず考えるべき中心テーマを
1つに絞ってください。

missing_information:
次の判断に影響する不足情報を
質問形式で最大3個まで示してください。

options:
現時点で考えられる選択肢を示してください。
断定ではなく、
検討可能な方向性として示してください。

next_actions:
利用者が次に行うとよいことを
1～3個程度で示してください。
"""


# =========================================================
# JSON抽出
# =========================================================

def extract_practice_json(text):
    """
    AI出力からJSONオブジェクトを抽出する。
    """

    if not text:
        raise ValueError(
            "AIから回答が返されませんでした。"
        )

    text = str(text).strip()

    # コードフェンスが付いた場合にも対応
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

        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Practice回答からJSONを"
            "取得できませんでした。"
        )

    json_text = text[
        start:end + 1
    ]

    return json.loads(
        json_text
    )


# =========================================================
# データ正規化
# =========================================================

def _to_list(value):
    """
    値を安全にリストへ変換する。
    """

    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    value = str(value).strip()

    if not value:
        return []

    return [value]


def normalize_practice_result(
    data,
):
    """
    AIから返されたPractice JSONを
    UIで扱いやすい形へ整える。
    """

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Practice回答の形式が"
            "正しくありません。"
        )

    scope = str(
        data.get(
            "scope",
            "個人",
        )
    ).strip()

    if scope not in PRACTICE_SCOPE_OPTIONS:
        scope = "個人"

    result = {
        "scope": scope,

        "contexts": _to_list(
            data.get(
                "contexts"
            )
        ),

        "important_now": str(
            data.get(
                "important_now",
                "",
            )
        ).strip(),

        "known_facts": _to_list(
            data.get(
                "known_facts"
            )
        ),

        "needs_organization": _to_list(
            data.get(
                "needs_organization"
            )
        ),

        "decision_focus": str(
            data.get(
                "decision_focus",
                "",
            )
        ).strip(),

        "missing_information": _to_list(
            data.get(
                "missing_information"
            )
        )[:3],

        "options": _to_list(
            data.get(
                "options"
            )
        ),

        "next_actions": _to_list(
            data.get(
                "next_actions"
            )
        )[:3],
    }

    return result


# =========================================================
# AI回答 → Practiceデータ
# =========================================================

def parse_practice_response(
    text,
):
    """
    AIの文字列回答を、
    Practice MVP用の辞書へ変換する。
    """

    data = extract_practice_json(
        text
    )

    return normalize_practice_result(
        data
    )