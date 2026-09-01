from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


LLM = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100, "temperature": 0.7}
    )

model = ChatHuggingFace(llm=LLM)
response = model.invoke("What is the capital for India?")   
print("script started")
print(response.content)
