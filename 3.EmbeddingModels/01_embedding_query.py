from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv  
load_dotenv()

model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
 
result = model.embed_query("hello world")

print(str(result))
