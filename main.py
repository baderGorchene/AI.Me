import os
import torch
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoTokenizer

# Model ID — Qwen2.5-0.5B-Instruct is small enough for CPU inference
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# Global variables
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model, tokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # Load model — use float32 on CPU for compatibility
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    yield

    model = None
    tokenizer = None


app = FastAPI(
    title="AI.Me — Qwen 2.5 0.5B API",
    description="A single FastAPI endpoint for text generation using transformers",
    version="1.0.0",
    lifespan=lifespan,
)


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class PromptResponse(BaseModel):
    response: str


@app.post("/generate", response_model=PromptResponse)
async def generate_text(request: PromptRequest):
    """Generate text using the transformers model."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    try:
        # Tokenize the input prompt
        inputs = tokenizer(request.prompt, return_tensors="pt")

        # Generate with the model
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
            )

        # Decode only the generated tokens (skip the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return PromptResponse(response=text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
