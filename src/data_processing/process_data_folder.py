import re
import os

# Define the folder containing the WhatsApp data
input_folder = "data/emily/discord_messages"  # Replace this with your folder path
output_folder = "data/emily/processed"  # Folder where the processed data will be saved

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through each file in the input folder
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    
    # Check if it's a text file
    if file_name.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = file.readlines()

        # Initialize variables for processing
        formatted_pairs = []
        current_prompt = ""
        current_response = ""
        last_speaker = None
        is_user_response = False
        contact_name = None

        for line in raw_data:
            # Skip system messages
            if "Messages and calls are end-to-end encrypted" in line or "is a contact." in line:
                continue

            # Use regex to separate timestamp, user, and message
            match = re.search(r"\] (.*?): (.*)", line)
            if match:
                speaker, message = match.groups()

                # If it's the first line, extract the contact's name (assuming it’s not Bruno Dumont)
                if contact_name is None and speaker != "Bruno Dumont":
                    contact_name = speaker

                # Check if the current speaker is the bot's persona
                if speaker != "Bruno Dumont":
                    # If we're switching from the other speaker to Bruno, save the prompt-response pair
                    if not is_user_response and current_prompt:
                        formatted_pairs.append(f"{current_prompt.strip()}\n{current_response.strip()}")
                        current_prompt = ""
                        current_response = ""

                    # Add the message to the response, with the speaker's name only at the start of a new group
                    if last_speaker != speaker:
                        current_response += f"{speaker}: {message}\n"
                    else:
                        current_response += f"{message}\n"

                    is_user_response = True
                else:
                    # If we're switching from Bruno to the other speaker, save the prompt-response pair
                    if is_user_response and current_response:
                        formatted_pairs.append(f"{current_prompt.strip()}\n{current_response.strip()}")
                        current_prompt = ""
                        current_response = ""

                    # Add the message to the prompt, with the speaker's name only at the start of a new group
                    if last_speaker != speaker:
                        current_prompt += f"{speaker}: {message}\n"
                    else:
                        current_prompt += f"{message}\n"

                    is_user_response = False

                # Update the last speaker
                last_speaker = speaker

        # Save any remaining prompt-response pair
        if current_prompt and current_response:
            formatted_pairs.append(f"{current_prompt.strip()}\n{current_response.strip()}")

        # Use the contact's name (if it's not None) to define the output file name
        if contact_name:
            output_file_name = f"{contact_name}_conversation.txt"
        else:
            output_file_name = "conversation_with_unknown_person.txt"  # Default fallback if no contact name is found

        output_file_path = os.path.join(output_folder, output_file_name)

        # Save formatted data to the output file with single-line separation between pairs
        with open(output_file_path, "w", encoding="utf-8") as out_file:
            out_file.write("\n".join(formatted_pairs))

        print(f"Processed {file_name} and saved to {output_file_name}")