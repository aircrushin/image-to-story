import gradio as gr
from transformers import pipeline, set_seed
from api_token import API_TOKEN
import requests

# Initialize the pipelines outside of the function to avoid reloading the model every time
image_to_text_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def img2text(url):
    text = image_to_text_pipeline(url)[0]["generated_text"]
    print(text)
    return text

def generate_story(scenario):
    API_URL = 'https://model-app-func-modelscd-bf-ebf-ijqperximb.cn-shanghai.fcapp.run/invoke'
    payload = {
        "input": {
            "messages": [
                {"content": "Hello! 你是谁？", "role": "user"},
                {"content": "我是你的助手", "role": "assistant"},
                {"content": "请用英语把这句话改编成一个有意义的小故事：{scenario} ", "role": "user"}
            ]
        },
        "parameters": {"do_sample": True, "max_length": 512}
    }
    response = requests.post(API_URL, json=payload).json()
    story = response["Data"]["message"]["content"]
    print(story)
    return story

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    API_TOKEN = "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"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    payload = {"inputs": message}
    response = requests.post(API_URL, headers=headers, json=payload)
    print("task is done!")
    return response.content

def process_image_to_speech(url):
    text_description = img2text(url)
    story = generate_story(text_description)
    speech_audio = text2speech(story)
    return speech_audio

# Define the input and output components for Gradio interface
image_input = gr.inputs.Image(label="Upload Image", type="pil")
speech_output = gr.outputs.Audio(label="Speech Audio", type="numpy")

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image_to_speech,
    inputs=image_input,
    outputs=speech_output,
    title="Image to Story to Speech",
    description="Upload an image to generate a story and convert it to speech."
)

# Launch the interface
iface.launch()

