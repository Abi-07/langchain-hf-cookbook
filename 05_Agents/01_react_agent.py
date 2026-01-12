from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent 

@tool
def add_tool(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def hello_tool(name: str) -> str:
    """Greet a person by their name."""
    return f"Hello, {name}!"

tools = [add_tool, hello_tool]

# Initialize model
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.1}
)
chat_model = ChatHuggingFace(llm=llm)

# SUCCESS: This version handles the prompt internally
agent_executor = create_react_agent(chat_model, tools)

response = agent_executor.invoke({"messages": [("user", "Add 7 and 5 then say hello to John")]})
print(response["messages"][-1].content)