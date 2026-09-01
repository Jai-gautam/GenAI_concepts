from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation")

model = ChatHuggingFace(llm=LLM)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is the capital of France?")]

response = model.invoke(messages)

messages.append(AIMessage(content=response.content))

print(messages)

