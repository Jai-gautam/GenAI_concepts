from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()


prompt = PromptTemplate(
    template = 'provide me five facts on the topic {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({'topic':"black holes"})

print(response)



