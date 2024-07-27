# master_script.py
import subprocess

# List of scripts to run
scripts = ["gpt4o_evaluations.py", "huggface_api.py"]

for script in scripts:
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(f"Running {script}...\n{result.stdout}\n{result.stderr}")
