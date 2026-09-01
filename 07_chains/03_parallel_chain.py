from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

llm3 = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)
model3 = ChatHuggingFace(llm=llm3)

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template = 'from the given {text} provide me 5 question and answers',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'from the given {text} provide me 5 line summary',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'merge {question_answer} and {summary}and display them together',
    input_variables = ['question_answer','summary']
)


parallel_chain = RunnableParallel(
     {
        'question_answer': prompt1 | model1 | parser,
        'summary': prompt2 | model2 | parser
    }
)

final_chain = parallel_chain | prompt3 | model3 | parser

text = "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They are formed when massive stars collapse under their own gravity at the end of their life cycle. The boundary around a black hole is called the event horizon, beyond which nothing can return. Black holes can vary in size, from small ones with a few times the mass of our sun to supermassive black holes that reside at the centers of galaxies. They play a crucial role in the evolution of galaxies and can influence the formation of stars and planetary systems."

result = final_chain.invoke({'text':text})

print(result)