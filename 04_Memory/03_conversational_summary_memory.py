import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

# 1. Pipeline Setup
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
hf_pipeline = pipeline(
    task="text-generation",
    model=model_id,
    device="mps",
    torch_dtype=torch.bfloat16,
    max_new_tokens=100,
    return_full_text=False,
    do_sample=True,
    temperature=0.3
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 2. Chat History Store
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 3. Summarization Component
summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize the following chat concisely: {history}"
)
summarize_chain = summarize_prompt | llm | StrOutputParser()

def manage_memory(input_dict):
    session_id = input_dict["session_id"]
    history_obj = get_session_history(session_id)
    messages = history_obj.messages
    
    # If history is long (e.g., > 4 messages), summarize it to save space
    if len(messages) > 4:
        summary = summarize_chain.invoke({"history": messages})
        history_obj.clear()
        history_obj.add_ai_message(f"Previous chat summary: {summary.strip()}")
    
    return {"history": history_obj.messages, "input": input_dict["input"]}

# 4. Main Conversational Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the history provided."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Use RunnablePassthrough to pipe the memory management logic
chain = (
    RunnablePassthrough() 
    | manage_memory 
    | prompt 
    | llm 
    | StrOutputParser()
)

# 5. Execution Helper
def chat(session_id, text):
    response = chain.invoke({"input": text, "session_id": session_id})
    # Update the actual history store
    hist = get_session_history(session_id)
    hist.add_user_message(text)
    hist.add_ai_message(response.strip())
    return response.strip()

# Test cases
sid = "user_2025"
print(f"Response 1: {chat(sid, 'My name is Alice.')}")
print(f"Response 2: {chat(sid, 'I am from India.')}")
print(f"Response 3: {chat(sid, 'How many letters in my country name?')}")
print(f"Response 4: {chat(sid, 'What is my name?')}")