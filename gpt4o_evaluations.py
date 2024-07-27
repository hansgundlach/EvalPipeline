
import csv
from openai import OpenAI
# loading dataset
rows = []

with open('PredictionQuestions.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    headers = next(csv_reader)

    for row in csv_reader:
        rows.append(row)

# 2. Load model
client = OpenAI()
print(rows[1])
for row in rows:
    question  = row[0]
    prompt = f"please answer this question: {question}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=2000,
        )
        # print(url)
        print(response.choices[0].message.content)

        # Write the output to a file
        with open("Responses/gpt4o.txt", "a") as f:
            f.write(response.choices[0].message.content + "\n")
            f.write("\n\n")
    except Exception as e:
        print(e)
        print("Error")