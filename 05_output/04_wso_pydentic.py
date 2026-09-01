from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"  
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    title: str
    key_themes: Optional[list[str]] = Field(None, description="Key themes discussed in the review, if available")
    summary: Optional[str] = Field(None, description="Summary of the review, if available")
    rating:  Optional[float] = Field(None, description="Rating given in the review, if available")
    sentiment: Literal["+ve","-ve"] = Field( description="Sentiment of the review, either good (+ve) or bad (-ve)")
    author: Optional[str] = Field(None, description="Author of the review, if available")

stru_model = model.with_structured_output(Review)


response = stru_model.invoke("""For years, the standard iPhone felt like a second-class citizen compared to the Pro. In 2026, that changed. The iPhone 17 is the first non-Pro model to feature the 120Hz ProMotion display, making scrolling and animations feel buttery smooth. Combined with the A19 chip and a jump to 256GB base storage, it’s the most "Pro" a standard iPhone has ever felt.

While it lacks the dedicated telephoto (zoom) lens of the 17 Pro Max, its new 48MP Ultra-wide camera and 18MP Center Stage selfie camera deliver stunning results. If you don't need professional-grade video features like ProRes, this is the phone to get""")
print(response)



"""there is a problem !!!, will resolve it later, openai support this we will solve the problem by output parser"""