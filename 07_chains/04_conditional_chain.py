from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal as literal
from dotenv import load_dotenv

load_dotenv()

class Sentiment(BaseModel):
    sentiment: literal['positive','negative'] = Field(description="sentiment of the feedback")

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)


parser1 = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Sentiment)

prompt1 = PromptTemplate(
    template = "from the given fedback determine the sentiment of {feedback},{format_instructions}",
    input_variables = ['feedback'],
    partial_variables = {'format_instructions': parser2.get_format_instructions()}
)


chain1 = prompt1 | model | parser2
feedback = "The product is really bad and I am not satisfied with it."

prompt2 = PromptTemplate(
    template = "reply for positive feedback {feedback}" 
)

prompt3 = PromptTemplate(
    template = "reply for negative feedback {feedback}"
)


chain2 = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not determine the sentiment")
)

final_chain = chain1 | chain2

result = final_chain.invoke({'feedback':feedback})

print(result)