import base64
import gradio as gr
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import os

# System prompts
calendar_prompt = """
Please analyze the labeled calendar image provided. The image contains UI elements with numeric labels and their corresponding text. Your task is to extract specific information about the calendar's current state.

Provide the following details in JSON format:
[
    "state": "<open or closed>",
    "currently_selected_month": "<month-name>",
    "currently_selected_month_id": "<label id of UI element which captures the currently selected month>",
    "currently_selected_year": "<year in 4 digits>",
    "currently_selected_year_id": "<label id of UI element which captures the currently selected year>",
    "currently_selected_day": "<day in the range of 1 - 31>",
    "currently_selected_day_id": "<label id of UI element which captures the currently selected day>",
    "decrease_button_element_id": "<label id of decrease button>",
    "increase_button_element_id": "<label id of increase button>"
]
"""

combobox_prompt = """
Please analyze the labeled image provided. The image contains comboboxes and icons with numeric labels and their corresponding text. Your task is to extract information about the comboboxes and icons.

Provide the following details in JSON format:
[
    "comboboxes": [
        {
            "label_id": "<label id>",
            "text": "<text inside combobox>"
        }
    ],
    "icons": [
        {
            "label_id": "<label id>",
            "icon_description": "<description of the icon>"
        }
    ]
]
"""

# Available models
available_models = [
    "llama3.2-vision:11b-instruct-q4_K_M",
    "llama3.2-vision:11b",
    "llava:latest",
    "qwen7b:latest"
]

# Default model
default_model = "llama3.2-vision:11b-instruct-q4_K_M"

# Initialize Qwen-7B
def initialize_qwen():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    )
    return tokenizer, model

# Model initializers
model_initializers = {
    "llama3.2-vision:11b-instruct-q4_K_M": lambda: ChatOllama(model="llama3.2-vision:11b-instruct-q4_K_M", temperature=0),
    "llama3.2-vision:11b": lambda: ChatOllama(model="llama3.2-vision:11b", temperature=0),
    "llava:latest": lambda: ChatOllama(model="llava:latest", temperature=0),
    "qwen7b:latest": initialize_qwen
}

# Convert image to Base64
def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Query Qwen-7B
def query_qwen(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Query LLM
def query_llm(image_base64, model_name, task_prompt):
    if model_name == "qwen7b:latest":
        tokenizer, model = model_initializers[model_name]()
        return query_qwen(task_prompt, tokenizer, model)
    else:
        llm = model_initializers[model_name]()
        def prompt_func(data):
            text = data["text"]
            image = data["image"]

            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",
            }

            content_parts = []

            text_part = {"type": "text", "text": text}

            content_parts.append(image_part)
            content_parts.append(text_part)

            return [HumanMessage(content=content_parts)]

        chain = prompt_func | llm | StrOutputParser()

        query_chain = chain.invoke(
            {"text": task_prompt, "image": image_base64}
        )

        return query_chain

# Process calendar image
def process_calendar(image, model_name):
    image_base64 = convert_to_base64(image)
    description = query_llm(image_base64, model_name, calendar_prompt)
    return description

# Process combobox and icon image
def process_combobox_icons(image, model_name, user_prompt):
    image_base64 = convert_to_base64(image)
    combined_prompt = f"{combobox_prompt}\n{user_prompt}"
    description = query_llm(image_base64, model_name, combined_prompt)
    return description

# Main decision page
def main_page(decision):
    if decision == "Parse Calendar":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

# Gradio interfaces
calendar_interface = gr.Interface(
    fn=process_calendar,
    inputs=[
        gr.Image(type="pil"),
        gr.Radio(choices=available_models, label="Select Model", value=default_model)
    ],
    outputs="text",
    title="Calendar Parser",
    description="Upload a calendar image to parse its details."
)

combobox_interface = gr.Interface(
    fn=process_combobox_icons,
    inputs=[
        gr.Image(type="pil"),
        gr.Radio(choices=available_models, label="Select Model", value=default_model),
        gr.Textbox(label="Enter additional prompt for analysis", placeholder="Describe your task...")
    ],
    outputs="text",
    title="Combobox & Icon Parser",
    description="Upload an image with comboboxes and icons to extract their details."
)

# Main Gradio interface
with gr.Blocks() as main_interface:
    gr.Markdown("### Choose Parsing Task")
    decision = gr.Radio(["Parse Calendar", "Parse Combobox & Icons"], label="What would you like to do?")
    
    with gr.Row(visible=False) as calendar_page:
        calendar_interface.render()
    
    with gr.Row(visible=False) as combobox_page:
        combobox_interface.render()

    decision.change(
        main_page,
        inputs=[decision],
        outputs=[calendar_page, combobox_page]
    )

# Launch the interface
if __name__ == "__main__":
    main_interface.launch()
