from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import logging
from torch import cuda, backends


# Set up device
def get_device():
    return (
        "mps"
        if backends.mps.is_available()
        else "cuda" if cuda.is_available() else "cpu"
    )


def load_saved_model():
    device = get_device()
    print(f"Using device = {device}")


# Enable verbose logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Specify a new cache directory
os.environ["HF_HOME"] = "/home/gridsan/hgundlach/.cache/huggingface"
# Set lock timeout
os.environ["TRANSFORMERS_LOCK_TIMEOUT"] = "300"

# Specify the model name
model_name = "mistralai/Mistral-7B-v0.1"


# can you give this a progrss bar?

# Load model directly
# Download and save locally with authentication
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_api_key)
# can you give this a progrss bar?


# Save to local directory
local_dir = "./Models"
os.makedirs(local_dir, exist_ok=True)
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print(f"Model and tokenizer saved to {local_dir}")


# Function to generate text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids, max_length=max_length, num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Test the model with a sample prompt
prompt = "Once upon a time"
generated_text = generate_text(prompt)


os.makedirs("Responses", exist_ok=True)
# save generated text to a file
with open("Responses/mistral_device.txt", "w") as f:
    f.write(generated_text)
    f.write("\n\n")


print(generated_text)
