from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc = ["hello world", "hi there", "greetings", "good morning", "good evening", "good night"]
result = model.embed_documents(doc)

print(str(result))

