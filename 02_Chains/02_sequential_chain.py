from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

hf_pipeline = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
    max_new_tokens=150,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

keywords_template = "Extract 3-5 key technical terms from the following text:\n\n{text}"
keywords_prompt = PromptTemplate.from_template(template=keywords_template)
keyword_chain = keywords_prompt | llm | StrOutputParser()

summary_template = "Given these keywords: {keywords}\n\nWrite a one-sentence summary of this original text: {text}"
summary_prompt = PromptTemplate.from_template(template=summary_template)
summary_chain = summary_prompt | llm | StrOutputParser()

chain = (
    {"keywords": keyword_chain, "text": itemgetter("text")}
    | summary_chain | StrOutputParser()
)


input_text = "LangChain is a powerful framework for developing applications powered by language models. It provides modules for model I/O, retrieval, chains, agents, and memory, enabling developers to build robust RAG applications with ease."
result = chain.invoke({"text": input_text})
print(result)