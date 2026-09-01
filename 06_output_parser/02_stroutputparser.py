from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({"topic":"climate change"})

print(response)
