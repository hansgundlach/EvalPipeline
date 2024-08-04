from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import logging
from torch import cuda, backends
from tqdm.auto import tqdm

# Set up device
def get_device():
    return "mps" if backends.mps.is_available() else "cuda" if cuda.is_available() else "cpu"

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
model_name = "gpt2"

# Load model directly with a progress bar
# Download and save locally with authentication
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_api_key)

# Save to local directory with progress bars
local_dir = "./Models/gpt2"
os.makedirs(local_dir, exist_ok=True)

for obj, path in tqdm([(tokenizer, f"{local_dir}/tokenizer"), (model, f"{local_dir}/model")], desc="Saving model and tokenizer"):
    obj.save_pretrained(path)

print(f"Model and tokenizer saved to {local_dir}")

# Function to generate text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model with a sample prompt
prompt = "Once upon a time"
generated_text = generate_text(prompt)

# Save generated text to a file
os.makedirs("Responses", exist_ok=True)
with open("Responses/gpt2_device.txt", "w") as f:
    f.write(generated_text)
    f.write("\n\n")

print(generated_text)
