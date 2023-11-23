from transformers import pipeline, set_seed
import requests
#image to text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    print(text)
    return text

#message = img2text("pic.png")
#print(message)

def generate_story(scenario):
    API_URL = 'https://model-app-func-modelscd-bf-ebf-ijqperximb.cn-shanghai.fcapp.run/invoke'
    def post_request(url, json):
	    with requests.Session() as session:
		    response = session.post(url,json=json,)
		    return response
    payload = {"input":{"messages":[{"content":"Hello! 你是谁？","role":"user"},{"content":"我是你的助手","role":"assistant"},{"content":"请用英语把这句话改编成一个有意义的小故事：{scenario} ","role":"user"}]},"parameters":{"do_sample":True,"max_length":512}}
    response = post_request(API_URL, json=payload)
    #print("response:", response.json())
    response = response.json()
    story = response["Data"]["message"]["content"]
    print(story)
    return story


def text2speech(message):

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    API_TOKEN = "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"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    payload = {
        "inputs": message
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

#text2speech(message)

if __name__ == "__main__":
    message = img2text("pic.png")
    story = generate_story(message)
    text2speech(story)
    