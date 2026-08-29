from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()

# ここを実際のPDFファイル名に変更してください
file_path = "研究計画書.pdf"

# 1. PDFをOpenAIにアップロード
with open(file_path, "rb") as f:
    uploaded_file = client.files.create(
        file=f,
        purpose="assistants"
    )

print("File ID:")
print(uploaded_file.id)

# 2. Vector Storeを作成
vector_store = client.vector_stores.create(
    name="OT Research Documents"
)

print("\nVector Store ID:")
print(vector_store.id)

# 3. PDFをVector Storeに登録
result = client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=uploaded_file.id
)

print("\n登録しました。")
print(result)