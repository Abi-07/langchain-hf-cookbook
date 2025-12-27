from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env (e.g., HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# 1. Setup the Hugging Face Inference Endpoint
# We select a lightweight, instruction-tuned model from the Qwen 2.5 family
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    temperature=0.2, # Controls creativity: 0 is deterministic, 1 is highly creative
    max_new_tokens=512, # Maximum length of the generated response
)

# 2. Interface Wrapper
# This converts the standard LLM into a Chat Model interface, 
# allowing for structured message handling (System, Human, AI)
model = ChatHuggingFace(llm=llm)

# 3. Define the Prompt Structure
# We use a raw string to define how the LLM should perceive its persona and the user
prompt = """
You are a helpful AI assistant specializing in LangChain.
User: {input}
Assistant:
"""

# 4. Initialize the PromptTemplate
# This object handles the logic of injecting user variables into the string template
prompt_template = PromptTemplate(
    input_variables=["input"],
    template=prompt,
)

# 5. Format the Prompt with User Input
# Here we provide a specific question about LangChain to the prompt
user_input = "Explain the 5 core modules of LangChain (Model I/O, Retrieval, Chains, Agents, and Memory) and how they interact in a RAG application."
formatted_prompt = prompt_template.format(input=user_input)

# 6. Execute the Model Call
response = model.invoke(formatted_prompt)
print(response.content)