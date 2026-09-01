from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
load_dotenv()

st.header("Research Paper Summariser")

paper_title = st.selectbox(
    "Select a research paper:", ["men are brave", "karma is real", "god is one"])

style_type = st.selectbox(
    "Select a style:", ["formal", "informal", "poetic"])

length_type = st.selectbox(
    "Select a length:", ["short", "medium", "long"])

template = PromptTemplate(
    template = """Summarise the research paper titled '{paper_title}' in a {style_type} style and {length_type} length.""",
    input_variables = ["paper_title", "style_type", "length_type"],
    validate_template = True
)

def load_model():
    LLM = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
    )
    return ChatHuggingFace(llm=LLM)

model = load_model()



if st.button("Summarise"):
    prompt = template.invoke({
    "paper_title": paper_title,
    "style_type": style_type,
    "length_type": length_type
     })
    response = model.invoke(prompt)
    st.write(response.content)

