from vllm import LLM, SamplingParams

def test_vllm():
    print("Initializing vLLM...")
    try:
        # We will use a tiny model like Qwen 0.5B for a quick test so it doesn't download 10GB of weights if it's not downloaded
        llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", tensor_parallel_size=1)
        print("vLLM initialized successfully on GPU.")
        
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        
        print("Running inference test...")
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            
        print("Inference test passed.")
        
    except Exception as e:
        print(f"Error during vLLM test: {e}")

if __name__ == "__main__":
    test_vllm()
