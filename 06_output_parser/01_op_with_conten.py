from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"     
)
model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="provide me detail report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="provide me short summary on the following {content}",
    input_variables=['content']
)

prompt1 = template1.invoke({"topic":"climate change"})
response1 = model.invoke(prompt1)

prompt2 = template2.invoke({"content":response1.content})
response2 = model.invoke(prompt2)

print(response2.content)