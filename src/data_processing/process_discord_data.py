import os
import json
import re
import json

def clean_text(text):

    text = re.sub(r"<@\d+>", "[USER_MENTION]", text)
    text = re.sub(r"\\u[0-9a-fA-F]{4}", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def process_data(input_file, output_file):
    data = []
    
    with open(input_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            cleaned_line = clean_text(line)
            if cleaned_line and not cleaned_line.startswith('pls'):

                data.append({"text": cleaned_line})
    
    with open(output_file, "w") as jsonl_file:
        for entry in data:
            json.dump(entry, jsonl_file)
            jsonl_file.write("\n")

input_file = "IGPT/data/eric/discord_messages.txt" 
output_file = "IGPT/data/eric/cleaned_discord_messages.jsonl" 

def parse_text(file_path):
    extracted_text = []
    with open(file_path, 'r') as file:
        for line in file:
            text_entry = json.loads(line)
            extracted_text.append(text_entry["text"][1:-1])
    return extracted_text

process_data(input_file=input_file, output_file = output_file)


