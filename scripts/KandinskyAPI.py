import json
import os
import time
import base64
import requests
import argparse
from random import randint as r
from random import choice as ch

from tqdm import tqdm


class Text2ImageAPI:

    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_model(self):
        response = requests.get(self.URL + 'key/api/v1/models', headers=self.AUTH_HEADERS)
        data = response.json()
        return data[0]['id']

    def generate(self, prompt, model, images=1, width=512, height=512):
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "generateParams": {
                "query": f"{prompt}"
            }
        }

        data = {
            'model_id': (None, model),
            'params': (None, json.dumps(params), 'application/json')
        }
        response = requests.post(self.URL + 'key/api/v1/text2image/run', headers=self.AUTH_HEADERS, files=data)
        data = response.json()
        return data['uuid']

    def check_generation(self, request_id, attempts=10, delay=10):
        while attempts > 0:
            response = requests.get(self.URL + 'key/api/v1/text2image/status/' + request_id, headers=self.AUTH_HEADERS)
            data = response.json()
            if data['status'] == 'DONE':
                return data['images']

            attempts -= 1
            time.sleep(delay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loading images from the API')
    parser.add_argument('--prompt', type=str, default="people_photos", help='Prompt for save generation')
    parser.add_argument('--countImage', type=int, default=500, help='Number of generations')
    parser.add_argument('--path', type=str, default='D:/Datasets/Kandinsry/test', help='Path to save the image')

    prompts = {
        "cars": [
            "Generate an image of an automobile, a motorized vehicle designed for transportation, typically with four wheels, powered by an internal combustion engine, and capable of carrying passengers or cargo.",
            "Create a photograph of a car, a mode of transportation characterized by its ability to travel on roads, highways, and other paved surfaces, providing individuals with mobility and convenience.",
            "Generate an image of a motor vehicle designed for various purposes, including commuting, leisure, business, and emergency services, serving as a versatile tool for daily life.",
            "Create a picture of a car, an embodiment of human ingenuity and innovation, evolving over time to meet changing needs and preferences, from classic models to cutting-edge designs.",
            "Create a photograph of a vehicle known for its utility and convenience, providing individuals with mobility and accessibility to destinations near and far, enhancing quality of life and societal connectivity.",
            "Generate an image of a car that embodies the concept of a modern means of transportation, equipped with wheels for movement on the earth's surface. The car is a mechanical vehicle typically featuring four wheels, an engine, and a body designed for transporting passengers or cargo. It usually includes seats for the driver and passengers, a steering wheel for control, and safety systems such as airbags. The image should reflect the dynamic nature of the car, its stylish design, distinctive features, and characteristics such as headlights, bumpers, windows, and wheels. It is also important to consider the surrounding environment and context in which the car is presented to create an atmosphere and convey its purpose and functionality."
            ],
        "illustrations": [
            "Generate a dataset of illustrations featuring various landscapes and environments.",
            "Create images of characters from different cultures and time periods.",
            "Generate a variety of abstract artworks with unique shapes and colors.",
            "Create illustrations of animals in different habitats and poses.",
            "Generate caricatures of famous personalities and fictional characters.",
            "Create fantastical dreamscapes with surreal elements and vibrant colors.",
            "Generate posters related to economics, medicine, politics, and other topics.",
            "Create illustrations depicting historical events and figures.",
            "Generate digital paintings of futuristic cityscapes and technology.",
            "Create whimsical illustrations of everyday objects in unusual situations."
        ],
        "people_photos": [
            # Описания для мужчин
            "Create a captivating close-up portrait of a man, showcasing the intricate details of his face and allowing the viewer to delve into his unique personality and character. Ensure the image is in color, and consider the possibility of him wearing hats, glasses, or other accessories that complement his individual style.",
            "Generate an engaging close-up portrait of a man, where every line and contour of his face tells a story of his inner world. Let the viewer witness the depth of his emotions, aspirations, and dreams reflected in his eyes and expression. Ensure the image is in color, and take into account any accessories that add to the narrative of the portrait.",
            "Capture a captivating close-up portrait of a man, highlighting his distinct charm and individuality. Invite the observer to admire the complexity of his character and the intensity of his emotions frozen in time. Ensure the portrayal is in vibrant color, considering any accessories that add to his magnetic presence.",
            "Generate a mesmerizing close-up portrait of a man, celebrating his unique beauty and individuality. Let the viewer appreciate the richness of his character and the depth of his emotions, captured in the moment of the shot. Ensure the image is in color, and take into account any accessories that enhance his aura and presence.",
                
            # Описания для женщин
            "Generate a captivating close-up portrait of a woman, revealing the intricate details of her face and allowing the viewer to delve into her unique personality and character. Ensure the image is in color, and consider the possibility of her wearing hats, glasses, or other accessories that complement her individual style.",
            "Create an engaging close-up portrait of a woman, where every line and contour of her face tells a story of her inner world. Let the viewer witness the depth of her emotions, aspirations, and dreams reflected in her eyes and expression. Ensure the image is in color, and take into account any accessories that add to the narrative of the portrait.",
            "Create a mesmerizing close-up portrait of a woman, celebrating her unique beauty and individuality. Let the viewer appreciate the richness of her character and the depth of her emotions, captured in the moment of the shot. Ensure the image is in color, and take into account any accessories that enhance her aura and presence.",
        ]
    }

    args = parser.parse_args()
    prompt = prompts[args.prompt]

    

    api = Text2ImageAPI('https://api-key.fusionbrain.ai/', 
                        'F7876D948A522A4606768F29453649B5', 
                        'BF71464A00EE0C9F58A6C3AADF51DFBF')
    model_id = api.get_model()
    for i in tqdm(range(args.countImage)):
        uuid = api.generate(prompt[i%len(prompt)], model_id)
        images = api.check_generation(uuid)
        if images != None:
            image_data = base64.b64decode(images[0])
            filename = f"{args.path}/{i%len(prompt)}_{args.prompt.split('.')[0]} _ {r(0, 100000)}.jpg"
            while os.path.isfile(filename):
                filename = f"{args.path}/{args.prompt.split('.')[0]} _ {r(0, 100000)}.jpg"
            with open(filename, "wb+") as file:
                file.write(image_data)
