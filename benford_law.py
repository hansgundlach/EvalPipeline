# %%
import openai
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os


# Initialize the OpenAI API key
# load open ai api key frome env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# %%


def generate_numbers(prompt, n):
    """Generate a list of numbers using OpenAI GPT API."""
    numbers = []
    for _ in range(n):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.5,
        )
        text = response.choices[0].message["content"].strip()
        numbers.extend(re.findall(r"\b\d+\b", text))
    return [int(num) for num in numbers]


def leading_digit_distribution(numbers):
    """Calculate the leading digit distribution of a list of numbers."""
    leading_digits = [str(num)[0] for num in numbers]
    digit_counts = Counter(leading_digits)
    total_count = sum(digit_counts.values())
    distribution = {digit: count / total_count for digit, count in digit_counts.items()}
    return distribution


def plot_distribution(distribution):
    """Plot the distribution of leading digits compared to Benford's Law."""
    benford_distribution = {str(d): np.log10(1 + 1 / d) for d in range(1, 10)}

    digits = list(benford_distribution.keys())
    benford_probs = [benford_distribution[d] for d in digits]
    observed_probs = [distribution.get(d, 0) for d in digits]

    x = np.arange(len(digits))

    fig, ax = plt.subplots()
    ax.bar(x - 0.2, benford_probs, 0.4, label="Benford")
    ax.bar(x + 0.2, observed_probs, 0.4, label="Observed")

    ax.set_xlabel("Leading Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Leading Digit Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.legend()

    plt.show()


# Generate numbers
prompt = "Generate the numerical value of a random physical constant"
numbers = generate_numbers(prompt, 20)

# %%

# Calculate the distribution of leading digits
distribution = leading_digit_distribution(numbers)

# Plot the distribution
plot_distribution(distribution)

# %%
