from dotenv import load_dotenv
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Setup the Model Pipeline (Local Hugging Face)
hf_pipeline = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device="mps", # Change to 0 for NVIDIA or "cpu"
    max_new_tokens=250,
    temperature=0.1,
    return_full_text=False,
    clean_up_tokenization_spaces=True
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 2. Setup Vector Store (Chroma)
# Ensure this matches the directory used in your ingestion script
persist_directory = "./db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings
)

# Convert vector store to a retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Setup the RAG Chain with LCEL
def format_docs(docs):
    """Formats retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

template = """<|im_start|>system
Answer the question based ONLY on the following context. If the answer is not in the context, say you don't know.<|im_end|>
<|im_start|>user
Context: {context}
Question: {question}
Answer:<|im_end|>
<|im_start|>assistant"""

prompt = ChatPromptTemplate.from_template(template)

# The Modern LCEL Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Execution
if __name__ == "__main__":
    query = "What are the components of RAG?"
    print(f"Querying: {query}\n" + "-"*30)
    
    # In LCEL, .invoke() is the standard entry point
    response = rag_chain.invoke(query)
    
    print(f"Response: {response.strip()}")