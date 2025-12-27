from dotenv import load_dotenv
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# Load environment variables from .env (e.g., HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# 2. Specify the model 'weights' to download from Hugging Face
# Qwen 2.5 1.5B is a small but powerful model suitable for testing
model_id = "Qwen/Qwen2.5-1.5B-Instruct"


# 3. Create a Hugging Face 'Pipeline'
# This is a wrapper that handles Tokenization (text to numbers) and the Model itself
hf_pipeline = pipeline(
    task="text-generation",
    model=model_id,
    device="mps", # Use "cuda" for NVIDIA GPUs, "mps" for Apple Silicon, or omit for CPU
    torch_dtype=torch.bfloat16, # bfloat16 reduces memory usage by 50% without losing much intelligence
    max_new_tokens=150, # Limits the response length to save compute time
    return_full_text=False, # Setting this to False means we only get the AI's answer, not our original prompt
    do_sample=False # do_sample=False makes the model 'Greedy' (always picks the most likely word)
)

# 4. Wrap the local pipeline into LangChain's HuggingFacePipeline interface
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 5. Run the model
response = llm.invoke(
    "Explain Quantum Physics to a five-year-old.",
    max_new_tokens=100,
    temperature=0.7
)

print(response)