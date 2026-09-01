from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template ='provide me detailed report on the topic {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'based on the detailed report provided, give 5 important facts from the {text}',
    input_variables = ['text']
)

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':"black holes"})

print(result)