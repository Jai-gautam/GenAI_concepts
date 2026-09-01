from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

tamplate = ChatPromptTemplate([
    ("system", "You are a customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "What is the status of my order?")    
])

chat_history = []

with open("09_chathistory.txt", "r") as file:
    chat_history.extend(file.readlines())



prompt = tamplate.invoke({
    "chat_history": chat_history
})

print(prompt)