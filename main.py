import os
import uuid
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

# Definition of the model ID
# Qwen3.5-2B is NOT supported by vLLM v0.8.5 (missing Qwen3_5ForConditionalGeneration)
# Using Qwen2.5-0.5B-Instruct which is natively supported
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# Global variables
engine: AsyncLLMEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global engine
    
    # Configure the asyncio engine for vLLM
    # Adding trust_remote_code=True for Qwen compatibility as is often required
    # enable_prefix_caching=True is highly recommended for vLLM efficiency
    engine_args = AsyncEngineArgs(
        model=MODEL_ID,
        trust_remote_code=True,
        enable_prefix_caching=True
    )
    
    # Initialize the vLLM asynchronous engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # FastAPI is now ready
    yield
    
    # No explicit cleanup is strictly necessary as vLLM process handles exit
    engine = None

app = FastAPI(
    title="vLLM Qwen 3.5 2B API",
    description="A single FastAPI endpoint for generation using vLLM and Qwen 3.5 2B",
    version="1.0.0",
    lifespan=lifespan
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
    """
    Generate text using the vLLM engine.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="vLLM engine is not initialized.")

    # Unique request ID required by vLLM
    request_id = str(uuid.uuid4())

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    try:
        # Generate text using vLLM's asynchronous engine
        results_generator = engine.generate(
            prompt=request.prompt,
            sampling_params=sampling_params,
            request_id=request_id
        )

        final_output = None
        # Consume the generator to get the final completion result
        async for request_output in results_generator:
            final_output = request_output

        # Extract the generated text from the final output
        if final_output is not None and final_output.outputs:
            text = final_output.outputs[0].text
            return PromptResponse(response=text)

        raise HTTPException(status_code=500, detail="Model returned an empty response.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# This ensures that we act appropriately to standard run commands but the user 
# specified the execution command should be `uvicorn main:app`
