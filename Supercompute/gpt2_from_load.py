# use_model_offline.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


def load_model_and_tokenizer(save_directory="../Models/local_gpt2"):
    # Load the tokenizer from the saved directory
    tokenizer = GPT2Tokenizer.from_pretrained(save_directory)

    # Load the model from the saved directory
    model = GPT2LMHeadModel.from_pretrained(save_directory)

    return tokenizer, model


def generate_text(prompt, tokenizer, model, max_length=50):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using the model
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)

    # Decode the generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text


if __name__ == "__main__":
    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Define the input prompt
    prompt = "Once upon a time"

    # Generate text
    generated_text = generate_text(prompt, tokenizer, model)

    # Print the generated text
    print(generated_text)
