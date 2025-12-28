from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup the Model Pipeline
hf_pipeline = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
    max_new_tokens=250,
    temperature=0.1, # Lowered for more factual RAG responses
    return_full_text=False,
    clean_up_tokenization_spaces=True
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 2. Setup Vector Store
texts = [
    "The company remote policy allows working from anywhere in the world for 3 months a year.",
    "Employees must be online during core hours: 10 AM to 3 PM EST.",
    "Health benefits include full dental and vision coverage starting Day 1."
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 3. Setup the RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Using a clear instruction format
template = """<|im_start|>system
Answer the question based ONLY on the following context. If the answer is not in the context, say you don't know.<|im_end|>
<|im_start|>user
Context: {context}
Question: {question}
Answer:<|im_end|>
<|im_start|>assistant"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Execution
result = rag_chain.invoke("What is the company's remote policy?")
print(f"Result: {result.strip()}")