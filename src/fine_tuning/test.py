import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_path = "IGPT/fine_tune/saved_model_lora"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    input_list = [
        "input: who are you?, output: ", 
        "input: hi, how are you doing?, output: "]
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
