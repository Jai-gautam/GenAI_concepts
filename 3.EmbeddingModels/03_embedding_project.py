from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc = ["JAI IS A STUDENT OF COMPUTER SCIENCE",
       "KANAK IS A FINAL YEAR STUDENT IN ABESIT",
       "JATIN IS INTRESTED IN GENAI AND REACT"]

query = 'tell me about jai'

doc_embedding = embedding.embed_documents(doc)
query_embedding = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], doc_embedding)[0]

score = similarity_scores.tolist()
amx = max(score)
index = score.index(amx)
print(f"Most similar document: {doc[index]}")



