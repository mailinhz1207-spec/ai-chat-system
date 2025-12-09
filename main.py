import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "AI is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a critical, objective AI that challenges beliefs."},
            {"role": "user", "content": req.message}
        ]
    )
    return {
        "reply": response.choices[0].message.content
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

