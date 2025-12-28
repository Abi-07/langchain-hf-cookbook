from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from operator import itemgetter

hf_pipeline = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
    max_new_tokens=150,
    temperature=0.5,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# --- Chain 1: Sentiment Analysis ---
sentiment_prompt = ChatPromptTemplate.from_template(
    "Analyze the sentiment of the following text. Respond with only one word: Positive, Negative, or Neutral.\n\nText: {text}"
)
sentiment_chain = sentiment_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip())

# --- Chain 2: Complexity Analysis ---
complexity_prompt = ChatPromptTemplate.from_template(
    "Classify the technical complexity of this text. Respond with only one word: Beginner, Intermediate, or Advanced.\n\nText: {text}"
)
complexity_chain = complexity_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip())

# --- Step 3: The Parallel Map ---
# RunnableParallel runs all branches in parallel and returns a dictionary
parallel_analysis = RunnableParallel(
    sentiment=sentiment_chain,
    complexity=complexity_chain,
    # original_text=itemgetter("text") # Carry the original text forward
)

# Execution
input_data = {"text": "The implementation of the transformer architecture revolutionized natural language processing by using self-attention mechanisms."}
result = parallel_analysis.invoke(input_data)

print(f"Analysis Results:\n{result}")