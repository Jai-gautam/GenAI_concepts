from langchain_google_genai import ChatGoogleGenAI
from dotenv import load_dotenv          
load_dotenv()
model = ChatGoogleGenAI(model="gemini-1.5-pro",temperature=0.7,max_completion_tokens=10)
response = model.invoke("What is the capital of France?")
print(response.content)

