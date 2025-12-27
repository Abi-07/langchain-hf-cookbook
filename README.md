# langchain-hf-cookbook

A comprehensive collection of LangChain implementations using Python and Hugging Face models. Includes examples of RAG pipelines, custom LLM integration, and autonomous agents.

LangChain + Hugging Face Mastery 🚀
A comprehensive guide to building LLM-powered applications using LangChain, Hugging Face, and Python. This repository covers everything from basic chains to advanced Retrieval-Augmented Generation (RAG) and autonomous agents using 2025's best practices.
🛠️ Tech Stack

# LangChain + Hugging Face Cookbook 🚀

> A comprehensive guide to building LLM-powered applications using [LangChain](https://python.langchain.com/) and [Hugging Face](https://huggingface.co/) in Python. Includes examples of RAG pipelines, custom LLM integration, and autonomous agents, following 2025's best practices.

---

## 🛠️ Tech Stack

- **Orchestration:** LangChain (v1.2.0+)
- **Models:** Hugging Face Hub (Mistral, Llama 3.x, Falcon)
- **Embeddings:** sentence-transformers via Hugging Face
- **Language:** Python 3.10+
- **Environment:** Hugging Face Inference API or Local Transformers

## 📂 Repository Structure

```
langchain-hf-cookbook/
├── 01_Basics/           # LLM setups and Prompt Templates
├── 02_Chains/           # Simple, Sequential, and Transform chains
├── 03_RAG_Pipelines/    # Vector stores (Chroma/FAISS) and Document Loaders
├── 04_Memory/           # ConversationBuffer and Summary Memory
├── 05_Agents/           # Tools and Zero-shot React Agents
├── requirements.txt     # Project dependencies
└── .env.example         # Template for API keys
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Abi-07/langchain-hf-cookbook.git
cd langchain-hf-cookbook
```

### 2. Set up Environment

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API Keys

To use Hugging Face models, generate a token at [Hugging Face Settings](https://huggingface.co/settings/tokens).
Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## 💡 Key Concepts Covered

- **Model I/O:** Interfacing with open-source LLMs via `HuggingFaceEndpoint` and `HuggingFacePipeline`.
- **Retrieval:** Building a knowledge base with `HuggingFaceEmbeddings`.
- **Prompt Engineering:** Creating dynamic templates for specific tasks.
- **Agents:** Giving LLMs the ability to browse the web or perform calculations.

---

## 📝 Usage Example

```python
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
   repo_id="Qwen/Qwen2.5-1.5B-Instruct",
   task="text-generation",
)

response = llm.invoke("What are the benefits of LangChain?")
print(response)
```

---

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request or open an issue for any new concepts you'd like to see added.

## 📄 License

This project is licensed under the MIT License.

🚀 Getting Started

1. Clone the repository
   bash
   git clone github.com
   cd langchain-huggingface-hub
   Use code with caution.

2. Set up Environment
   Create a virtual environment and install dependencies:
   bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   Use code with caution.

3. Configure API Keys
   To use Hugging Face models, generate a token at Hugging Face Settings.
   Create a .env file:
   text
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   Use code with caution.

💡 Key Concepts Covered
Model I/O: Interfacing with Open-Source LLMs via HuggingFaceEndpoint and HuggingFacePipeline.
Retrieval: Building a knowledge base with HuggingFaceEmbeddings.
Prompt Engineering: Creating dynamic templates for specific tasks.
Agents: Giving LLMs the ability to browse the web or perform calculations.
📝 Usage Example
python
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
repo_id="Qwen/Qwen2.5-1.5B-Instruct",
task="text-generation",
)

response = llm.invoke("What are the benefits of LangChain?")
print(response)
Use code with caution.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any new concepts you'd like to see added.
📄 License
This project is licensed under the MIT License.
