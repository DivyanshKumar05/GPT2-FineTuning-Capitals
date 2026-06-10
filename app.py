import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup the device to run on GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Your specific Hugging Face model repository
model_id = "DIvyansh1929/gpt2-finetuned-capitals"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

def generate_text(prompt):
    # Prepare the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate the output using Greedy Decoding (factual, no randomness)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=15,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Build the Gradio web interface
interface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Question: What is the capital of Japan?\nAnswer:"),
    outputs=gr.Textbox(label="Model Output"),
    title="Fine-Tuned GPT-2 Geography Expert",
    description="A custom-tuned GPT-2 model specialized in delivering factual country capitals."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()