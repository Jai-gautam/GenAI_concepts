from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",      
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Vehical(BaseModel):
    name: str = Field(description="name of the car")
    reg_num: int = Field(gt=10, description="registration number of the car")
    color: str = Field(description="color of the car")
    owner: str = Field(description="name of the person who owns the car")

parser = PydanticOutputParser(pydantic_object=Vehical)

template = PromptTemplate(
    template="provide me info a car of {brand} \n {format_instruction}",
    input_variables=['brand'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
 
)
prompt = template.invoke({'brand':"BYD"})
response = model.invoke(prompt)

result = parser.parse(response.content)
print("promt starts")
print(prompt)
print("prompt ends")
print(result)

