import re

# Load raw WhatsApp data
with open("whatsapp_data.txt", "r", encoding="utf-8") as file:
    raw_data = file.readlines()

# Initialize variables for processing
formatted_pairs = []
current_prompt = ""
current_response = ""
last_speaker = None
is_user_response = False

for line in raw_data:
    # Skip system messages
    if "Messages and calls are end-to-end encrypted" in line or "is a contact." in line:
        continue
    
    # Use regex to separate timestamp, user, and message
    match = re.search(r"\] (.*?): (.*)", line)
    if match:
        speaker, message = match.groups()
        
        # Check if current speaker is Bruno (the bot's persona)
        if speaker != "Bruno Dumont":
            # If we're switching from the other speaker to Bruno, save the prompt-response pair
            if not is_user_response and current_prompt:
                formatted_pairs.append(f"{current_prompt.strip()}\n{current_response.strip()}")
                current_prompt = ""
                current_response = ""
            
            # Add the message to the response, with speaker's name only at the start of a new group
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
            
            # Add the message to the prompt, with speaker's name only at the start of a new group
            if last_speaker != speaker:
                current_prompt += f"{speaker}: {message}\n"
            else:
                current_prompt += f"{message}\n"
            
            is_user_response = False

        # Update last_speaker to current speaker
        last_speaker = speaker

# Save any remaining prompt-response pair
if current_prompt and current_response:
    formatted_pairs.append(f"{current_prompt.strip()}\n{current_response.strip()}")

# Save formatted data to a text file with single-line separation between pairs
with open("prompt_response_pairs.txt", "w", encoding="utf-8") as out_file:
    out_file.write("\n".join(formatted_pairs))