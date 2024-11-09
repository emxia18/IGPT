import os
import json
import re
import json

# # Specify the root directory where the folders are stored
# root_dir = "IGPT/data/eric/messages"
# output_file = "IGPT/data/eric/discord_messages.txt"

# # Create or clear the output file
# with open(output_file, 'w') as outfile:
#     pass

# # Traverse the directories
# for folder in os.listdir(root_dir):
#     folder_path = os.path.join(root_dir, folder)
#     if os.path.isdir(folder_path):  # Ensure it's a folder
#         messages_file = os.path.join(folder_path, "messages.json")
#         if os.path.exists(messages_file):  # Check if the file exists
#             with open(messages_file, 'r') as infile:
#                 messages = json.load(infile)  # Load the messages
#                 with open(output_file, 'a') as outfile:
#                     for message in messages:
#                         if "Contents" in message:
#                             outfile.write(json.dumps(message["Contents"]) + "\n")  # Write only Contents field

# print(f"Combined messages saved to {output_file}")

def clean_text(text):
    """
    Cleans a single line of text by removing unwanted characters, mentions, and excess whitespace.
    """
    # Remove mentions (e.g., <@123456789>)
    text = re.sub(r"<@\d+>", "[USER_MENTION]", text)
    # Remove emojis and unicode characters (e.g., \ud83d\ude4f)
    text = re.sub(r"\\u[0-9a-fA-F]{4}", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

def process_data(input_file, output_file):
    """
    Reads the raw data, cleans it, and saves it in a structured JSONL format.
    """
    data = []
    
    with open(input_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            # Clean each line
            cleaned_line = clean_text(line)
            # Skip empty lines
            if cleaned_line and not cleaned_line.startswith('pls'):

                data.append({"text": cleaned_line})
    
    # Save to JSONL format
    with open(output_file, "w") as jsonl_file:
        for entry in data:
            json.dump(entry, jsonl_file)
            jsonl_file.write("\n")

# Specify file paths
input_file = "IGPT/data/eric/discord_messages.txt"  # Replace with your input file path
output_file = "IGPT/data/eric/cleaned_discord_messages.jsonl"  # Replace with your desired output file path

# Process the data
# process_data(input_file, output_file)
def parse_text(file_path):
    extracted_text = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as JSON
            text_entry = json.loads(line)
            # Remove quotes from the beginning and end of the text
            extracted_text.append(text_entry["text"][1:-1])
    return extracted_text

process_data(input_file=input_file, output_file = output_file)
# input_file_path = 'IGPT/data/eric/cleaned_discord_messages.jsonl'

# # Output file path
# output_file_path = 'IGPT/Bayes Model Eric/eric_messages.txt'

# # Parse the text from the input file
# parsed_text = parse_text(input_file_path)

# # Write the parsed text to the output file
# with open(output_file_path, 'w') as f:
#     for line in parsed_text:
#         f.write(line + '\n')

# print(f"Parsed text has been written to {output_file_path}")


