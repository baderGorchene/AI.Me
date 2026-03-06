import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import AsyncGroq
from pydantic import BaseModel

load_dotenv()

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Best free-tier model: 14,400 req/day, 30 req/min, 6K tokens/min
MODEL_ID = os.getenv("MODEL_ID", "llama-3.1-8b-instant")

# Global Groq client
client: AsyncGroq | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global client

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Get one free at https://console.groq.com/keys"
        )

    client = AsyncGroq(api_key=GROQ_API_KEY)

    yield

    client = None


app = FastAPI(
    title="AI.Me",
    description="Personal AI assistant powered by Groq",
    version="1.0.0",
    lifespan=lifespan,
)


class PromptRequest(BaseModel):
    text: str


class PromptResponse(BaseModel):
    response: str


@app.post("/generate", response_model=PromptResponse)
async def generate_text(request: PromptRequest):
    """Generate text using Groq's LLM API."""
    if client is None:
        raise HTTPException(status_code=500, detail="Groq client is not initialized.")

    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": request.text,
                }
            ],
            model=MODEL_ID,
            max_tokens=512,
            temperature=0.7,
        )

        text = chat_completion.choices[0].message.content
        return PromptResponse(response=text or "")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
