from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load Documents from the 'data' folder
loader = DirectoryLoader("./data", glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

# 2. Split into chunks 
# Recursive splitter tries to keep paragraphs/sentences together
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
docs = text_splitter.split_documents(documents)

# 3. Create Local Embeddings using Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in ChromaDB
persist_directory = "./db"
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"Success: Vector store saved to {persist_directory}")