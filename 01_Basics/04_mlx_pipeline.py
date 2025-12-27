import mlx_lm
from langchain_mlx import MLXPipeline

# 1. Load the model and tokenizer
# This happens locally on your Mac's GPU (Unified Memory)
model_id = "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
model, tokenizer = mlx_lm.load(model_id)

# 2. Wrap in the LangChain Pipeline
llm = MLXPipeline(
    model=model,
    tokenizer=tokenizer,
    pipeline_kwargs={"max_tokens": 150, "temp": 0.7}
)

# 3. Simple execution (Direct string input, no chains/documents)
question = "Explain airplane to a five-year-old."

# Mistral models usually expect instruction tags [INST] [/INST]
formatted_prompt = f"<s>[INST] {question} [/INST]"

response = llm.invoke(formatted_prompt)

print(response)