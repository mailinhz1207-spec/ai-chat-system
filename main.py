# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid

from assistant import Assistant
from memory_store import MemoryStore
from policy import PolicyEngine

# load env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment")

app = FastAPI(title="AI Chat System - Stable Advanced")

# singletons (for simplicity)
memory = MemoryStore(namespace="global")   # use sqlite+faiss or pinecone inside implementation
assistant = Assistant(openai_api_key=OPENAI_API_KEY, memory=memory)
policy = PolicyEngine()

class MessageIn(BaseModel):
    user_id: str
    text: str
    session_id: str | None = None

class MessageOut(BaseModel):
    reply: str
    metadata: dict

@app.post("/chat", response_model=MessageOut)
async def chat(msg: MessageIn):
    # policy check
    if not policy.allow_message(msg.user_id, msg.text):
        raise HTTPException(status_code=400, detail="Message violates policy or age-safety rules.")

    # retrieve relevant memories
    relevant = memory.retrieve(user_id=msg.user_id, query=msg.text, k=6)

    # get assistant reply (this calls OpenAI)
    reply, mem_to_commit = await assistant.generate_reply(
        user_id=msg.user_id,
        user_text=msg.text,
        session_id=msg.session_id,
        context_memories=relevant
    )

    # persist selected memory items (only non-sensitive, filtered by policy)
    for m in mem_to_commit:
        memory.add_memory(user_id=msg.user_id, content=m["content"], metadata=m.get("meta", {}))

    return MessageOut(reply=reply, metadata={"memories_used": len(relevant)})

@app.post("/sync_memory/{user_id}")
def force_condense(user_id: str):
    # force a condense/summary pass for a user (run occasionally)
    memory.condense(user_id=user_id)
    return {"status": "condensed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
