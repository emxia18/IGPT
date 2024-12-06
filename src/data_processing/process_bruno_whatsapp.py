import re
import os

input_folder = "IGPT/data/bruno/whatsapp_data_folder" 
output_folder = "IGPT/data/bruno/processed_data_folder"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    
    if file_name.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = file.readlines()

        formatted_pairs = []
        current_prompt = ""
        current_response = ""
        last_speaker = None
        is_user_response = False
        contact_name = None

        for line in raw_data:
            if "Messages and calls are end-to-end encrypted" in line or "is a contact." in line:
                continue

            match = re.search(r"\] (.*?): (.*)", line)
            if match:
                speaker, message = match.groups()

                if contact_name is None and speaker != "Bruno Dumont":
                    contact_name = speaker

                if speaker != "Bruno Dumont":
                    if not is_user_response and current_prompt:
                        formatted_pairs.append(f"{current_prompt.strip()}\n{current_response.strip()}")
                        current_prompt = ""
                        current_response = ""

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

                    if last_speaker != speaker:
                        current_prompt += f"{speaker}: {message}\n"
                    else:
                        current_prompt += f"{message}\n"

                    is_user_response = False

                last_speaker = speaker

        if current_prompt and current_response:
            formatted_pairs.append(f"{current_prompt.strip()}\n{current_response.strip()}")

        if contact_name:
            output_file_name = f"{contact_name}_conversation.txt"
        else:
            output_file_name = "conversation_with_unknown_person.txt" 

        output_file_path = os.path.join(output_folder, output_file_name)

        with open(output_file_path, "w", encoding="utf-8") as out_file:
            out_file.write("\n".join(formatted_pairs))

        print(f"Processed {file_name} and saved to {output_file_name}")