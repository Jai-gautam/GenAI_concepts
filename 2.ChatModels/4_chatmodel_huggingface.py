from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=LLM)
response = model.invoke("What is the capital for India?")
print(response.content)