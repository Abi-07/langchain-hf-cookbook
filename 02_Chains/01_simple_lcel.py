# LCEL-LangChain Expression Language

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

hf_pipeline = pipeline(
    task="text-generation",
    model = "Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
    max_new_tokens=150,
    temperature=0.2,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

template = "Write a short poem about {topic}."

prompt = PromptTemplate.from_template(template=template)

chain = prompt | llm

response = chain.invoke({"topic": "Ocean"})

print(response)
