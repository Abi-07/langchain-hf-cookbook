from dotenv import load_dotenv
from operator import itemgetter
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages

load_dotenv()

hf_pipeline = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
    max_new_tokens=150,
    temperature=0.1,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=hf_pipeline, pipeline_kwargs={"stop_strings": ["<|im_end|>", "<|im_start|>"]})

trimmer = trim_messages(
    max_tokens=1000, 
    strategy="last",
    token_counter=llm, # Using the LLM to count actual tokens
    include_system=True,
    start_on="human",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "<|im_start|>system\nYou are a factual assistant. Use ONLY the history below to answer. If not in history, say you don't know.<|im_end|>"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant"),
])

chain = (
    RunnablePassthrough.assign(history=itemgetter("history") | trimmer)
    | prompt
    | llm
    | StrOutputParser()
)

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

config = {"configurable": {"session_id": "user_2025"}}

print("Step 1:", with_message_history.invoke({"input": "Hi, my name is John Doe."}, config))
print("Step 2:", with_message_history.invoke({"input": "I am from Mexico."}, config))

resp3 = with_message_history.invoke({"input": "How many states in my country?"}, config)
print(f"States: {resp3}")

resp4 = with_message_history.invoke({"input": "What is my name?"}, config)
print(f"Name Check: {resp4}")