from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation")

model = ChatHuggingFace(llm=LLM)

chat_history = [SystemMessage(content="You are a helpful assistant")]
while True:
    user_input = input("Enter your query (or 'exit' to quit): ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Model response:", response.content)



print(chat_history)


