import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_path = "IGPT/fine_tune/saved_model_lora"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

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
                early_stopping=False,
                no_repeat_ngram_size=2,
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(output_text)

if __name__ == "__main__":
    main()
