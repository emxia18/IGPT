import os
import json

# Specify the root directory where the folders are stored
root_dir = "data/emily/discord_messages"
output_file = "data/emily/discord_messages.txt"

# Create or clear the output file
with open(output_file, 'w') as outfile:
    pass

# Traverse the directories
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):  # Ensure it's a folder
        messages_file = os.path.join(folder_path, "messages.json")
        if os.path.exists(messages_file):  # Check if the file exists
            with open(messages_file, 'r') as infile:
                messages = json.load(infile)  # Load the messages
                with open(output_file, 'a') as outfile:
                    for message in messages:
                        if "Contents" in message:
                            outfile.write(json.dumps(message["Contents"]) + "\n")  # Write only Contents field

print(f"Combined messages saved to {output_file}")
