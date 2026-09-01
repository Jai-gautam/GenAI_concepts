from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate([
    ("system", "You are a {domain} expert."),
    ("human", "tell me about {topic}")
])

prompt = template.invoke({
    "domain": "math",
    "topic": "calculus"
})

print(prompt)