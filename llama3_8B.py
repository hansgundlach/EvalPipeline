# %%
import csv
import requests

from dotenv import load_dotenv
import os
#%%
# Load dataset
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
rows = []
with open("PredictionQuestions.csv", mode="r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)

    for row in csv_reader:
        rows.append(row)

# Huggingface API setup
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B" 
headers = {"Authorization": f"Bearer {huggingface_api_key}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


print(rows[1])
for row in rows:
    question = row[0]
    prompt = f"please answer this question: {question}"

    try:
        response = query({"inputs": prompt})

        if "error" in response:
            print("Error:", response["error"])
        else:
            answer = response[0]["generated_text"]
            print(answer)

            # Write the output to a file
            with open("Responses/mistral_responses.txt", "a") as f:
                f.write(answer + "\n")
                f.write("\n\n")
    except Exception as e:
        print(e)
        print("Error")


# %%
