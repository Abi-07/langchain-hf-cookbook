import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
hf_pipeline = pipeline(
    task="text-generation",
    model=model_id,
    device="mps", 
    torch_dtype=torch.bfloat16,
    max_new_tokens=50,
    return_full_text=False,
    stop_sequence=["Human:", "\n\n"],
    do_sample=True,
    temperature=0.1
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Buffer Management (Sliding Window)
def manage_buffer(input_dict):
    session_id = input_dict["session_id"]
    history_obj = get_session_history(session_id)
    # Keep last 4 messages (2 rounds of conversation)
    return {"history": history_obj.messages[-4:], "input": input_dict["input"]}

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Give concise answers based on the history."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = (
    RunnablePassthrough() 
    | manage_buffer 
    | prompt 
    | llm 
    | StrOutputParser()
)

def chat(session_id, text):
    response = chain.invoke({"input": text, "session_id": session_id})
    cleaned_response = response.strip().split("Human:")[0].split("\n")[0]
    
    hist = get_session_history(session_id)
    hist.add_user_message(text)
    hist.add_ai_message(cleaned_response)
    return cleaned_response

sid = "user_2026"
print(f"Response 1: {chat(sid, 'My name is Alice.')}")
print(f"Response 2: {chat(sid, 'I am from India.')}")
print(f"Response 3: {chat(sid, 'How many letters in my country name?')}")
print(f"Response 4: {chat(sid, 'What is my name?')}")