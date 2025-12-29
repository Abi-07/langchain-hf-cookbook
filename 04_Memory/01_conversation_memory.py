from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

hf_pipeline = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
    max_new_tokens=250,
    temperature=0.2,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = ChatPromptTemplate.from_messages([
    ("system", "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant"),
])

chain = prompt | llm | StrOutputParser()

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "user1"}}
response = with_message_history.invoke(
    {"input": "Hi, my name is John Doe. Can you tell me a joke about programming?"},
    config=config
)
print(response)

response2 = with_message_history.invoke(
    {"input": "what is my name?"},
    config=config
)

print(response2)