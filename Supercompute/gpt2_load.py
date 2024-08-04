# download_model.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os


def download_model(model_name="gpt2", save_directory="../Models/local_gpt2"):
    # Create directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Download and save the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)

    # Download and save the model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(save_directory)

    print(f"Model and tokenizer saved to {save_directory}")


if __name__ == "__main__":
    download_model()
