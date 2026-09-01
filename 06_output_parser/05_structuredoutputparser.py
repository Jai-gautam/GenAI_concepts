from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="name",description="the name of the person"),   
    ResponseSchema(name="age",description="the age of the person"),
    ResponseSchema(name="hobby",description="the hobby of the person")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="provide me detail of a fictional person \n {formate_example}",
    input_variables=[],
    partial_variables= {"formate_example": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)
