import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

# 1. Initialize the pipeline
nlp = pipeline(
    task="fill-mask",
    model="distilbert-base-uncased"
)

# 2. Execute the pipeline
result = nlp("The quick brown fox jumps over the lazy [MASK].")

# 3. Print only the top predicted text (the completed sentence)
# 'result' is a list; we take the first element [0] as it has the highest score
print(result[0]['sequence'])