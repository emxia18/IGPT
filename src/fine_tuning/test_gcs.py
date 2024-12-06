import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from google.cloud import storage

bucket_name = "igpt_fine_tuned_models"
model_path = "llama-7b-emily/"
local_dir = "./saved_model/"

def download_model_from_gcs(bucket_name, model_path, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    os.makedirs(local_dir, exist_ok=True)

    blobs = bucket.list_blobs(prefix=model_path)

    for blob in blobs:
        local_file_path = os.path.join(local_dir, os.path.relpath(blob.name, model_path))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")

download_model_from_gcs(bucket_name, model_path, local_dir)

hf_token = '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'

model = AutoModelForCausalLM.from_pretrained(local_dir, use_auth_token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(local_dir, use_auth_token=hf_token)
model.eval()

input_list = [
    "input: How are you doing?, output: ",
    "input: What's your plans for the day?, output: ",
    "input: How was your week?, output: ",
    "input: Do you want to join me for dinner?, output: ",
    "input: Are you doing anything tomorrow afternoon?, output: ",
    "input: What are you thinking about right now?, output: ",
    "input: Do you feel like going out later?, output: ",
    "input: What’s been on your mind lately?, output: ",
    "input: How are you feeling today?, output: ",
    "input: Do you have any fun plans this weekend?, output: ",
    "input: What do you usually do in the mornings?, output: ",
    "input: Did you have a good day so far?, output: ",
    "input: What are you craving to eat right now?, output: ",
    "input: Do you want to hang out sometime?, output: ",
    "input: How do you usually spend your evenings?, output: ",
    "input: Did anything interesting happen today?, output: ",
    "input: Do you want to try something new this week?, output: ",
    "input: What’s your favorite thing to do on a lazy day?, output: ",
    "input: Are you free to chat for a bit?, output: ",
    "input: What’s something you’re excited about?, output: ",
    "input: How do you like to start your day?, output: ",
    "input: Is there anything you’re looking forward to?, output: ",
    "input: Do you need help with anything right now?, output: ",
    "input: What do you usually do for fun?, output: ",
    "input: Do you feel like going for a walk?, output: "
]

for input in input_list:
    input_ids = tokenizer.encode(input, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)