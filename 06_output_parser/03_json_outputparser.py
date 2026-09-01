from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",      
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="provide me detail of a fictional person /n {formate_example}",
    input_variables=[],
    partial_variables= {"formate_example": parser.get_format_instructions()}
)

prompt1 = template1.invoke({})
response1 = model.invoke(prompt1)
print(response1.content)
print(parser.parse(response1.content   ))