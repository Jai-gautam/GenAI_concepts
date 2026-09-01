from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.7,max_completion_tokens=10)
response = model.invoke("What is the capital of France?")
print(response.content)