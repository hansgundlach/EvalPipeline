#%%
import csv
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

#%%
# Load dataset
rows = []

with open('PredictionQuestions.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)

    for row in csv_reader:
        rows.append(row)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(rows[1])


#%%
for row in rows:
    question = row[0]
    prompt = f"Please answer this question: {question}"

    try:
        response = generate_response(prompt)
        print(response)

        # Write the output to a file
        with open("Responses/gpt2_responses.txt", "a") as f:
            f.write(response + "\n")
            f.write("\n\n")
    except Exception as e:
        print(e)
        print("Error")

# %%
