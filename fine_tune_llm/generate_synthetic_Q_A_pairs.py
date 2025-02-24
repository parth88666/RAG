import os
import json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Load API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY in .env")

# Path to text files
TXT_DIRECTORY = "./articles/"

# Load all summaries from .txt files
def load_summaries(directory):
    summaries = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                summaries.append({"source": file, "content": f.read()})
    return summaries

# Generate synthetic Q&A pairs from summaries
def generate_qa_pairs(summary_text):
    model = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_api_key)
    
    prompt = f"""
    Given the following news summary, generate 3 question-answer pairs:
    
    Summary: {summary_text}

    Format as a JSON list:
    [
      {{"question": "Generated question?", "context": "Relevant part of the summary", "answer": "Concise response"}}
    ]
    """
    
    response = model([HumanMessage(content=prompt)]).content
    return json.loads(response)

# Process all summaries
summaries = load_summaries(TXT_DIRECTORY)
qa_data = []

for summary in summaries:
    try:
        qa_pairs = generate_qa_pairs(summary["content"])
        for qa in qa_pairs:
            qa["source"] = summary["source"]  # Add source metadata
        qa_data.extend(qa_pairs)
    except Exception as e:
        print(f"Error processing {summary['source']}: {e}")

# Save structured Q&A dataset
output_file = "qa_dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(qa_data, f, indent=2)

print(f"Q&A dataset saved to {output_file}")

