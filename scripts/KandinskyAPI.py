import json
import os
import time
import base64
import requests
import argparse
from random import randint as r
from random import choice as ch
from requests.exceptions import HTTPError, ConnectionError, Timeout

from tqdm import tqdm


class Text2ImageAPI:

    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_model(self):
        while True:  # Повторять запрос, пока не будет получен успешный ответ
            try:
                response = requests.get(self.URL + 'key/api/v1/models', headers=self.AUTH_HEADERS)
                response.raise_for_status()  # Проверяем статус ответа, вызываем исключение, если не 2xx
                data = response.json()
                return data[0]['id']
            except requests.HTTPError as e:
                print(f"HTTP Error: {e}")
                print("Retrying...")
                time.sleep(100)  # Пауза перед повторным запросом
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                time.sleep(100)  # Пауза перед повторным запросом

    def generate(self, prompt, model, images=1, width=1024, height=1024):
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "generateParams": {
                "query": f"{prompt}"
            },
            "name":"UHD",
            "negativePromptUnclip": "bright colors, acidity, high contrast"
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
    parser.add_argument('--prompt', type=str, default="Landscape", help='Prompt for save generation')
    parser.add_argument('--countImage', type=int, default=1000, help='Number of generations')
    parser.add_argument('--path', type=str, default='D:/Datasets/Kandinsry/Landscapes', help='Path to save the image')

    prompts = {
        'Landscape':
        [
            # For the forest
            "A photorealistic forest showcasing the beauty of untouched nature. Leaves whisper the secrets of the wind, creating an atmosphere of mystery.",
            "A forest shrouded in mystical fog, where trees seem woven from the fabric of dreams. Sunbeams penetrate through the foliage, creating a play of light and shadow.",
            "A photorealistic forest where each tree appears as a guardian of ancient secrets. The rustling of leaves underfoot creates a sense of tranquility and seclusion.",
            
            # For the tundra
            "A photorealistic tundra where the sky merges with the earth as one. Frozen ground demonstrating the unchanging nature of its landscape.",
            "A photorealistic winter region, covered with a thin layer of snow like a white blanket. Icy formations give the impression of an undisturbed world.",
            "A harsh winter region, both frightening and mesmerizing with its constancy and cold. Mesmerizing variety of natural landscapes.",
            
            # For the seashore
            "Ocean shore where waves caress the beach, creating the music of nature.",
            "A seaside bathed in the setting sun, creating magical illumination. Palms lean towards the water, as if preparing to dive into the waves.",
            "A shoreline where the sea merges with the sky in a harmonious dance. Soft sand underfoot creates a feeling of calmness and serenity.",
            
            # For the cliffs
            "Cliffs rising above the sea like guardians protecting the shore from raging waves. Their stone walls seem eternal, unchanged over time.",
            "A rugged coastline where each rock looks like a work of art, carved by nature itself. Rocks of various shades create an impression of a colorful landscape.",
            "Cliffs covered with green ivy and vines, like green waves crashing onto the rocky shore. This place seems like a snippet from ancient legends.",
            "Cliffs washed by turquoise waves, creating an impression of unreality. Colorful seaweed and picturesque caves add mystical charm to this place.",
            "Cliffs, like gigantic statues, towering above the sea. Their contours create a play of light and shadow, as if nature has painted them on canvas with a brush.",
            
            # For mountain landscapes
            "Mountain peaks, tall and inaccessible, as if reaching for the sky. Rocks and glaciers give the impression of eternity and the power of nature.",
            "Photorealistic mountains where every rock and snowdrift looks so realistic that you can feel the cold wind on your face. Sun rays play on the slopes, creating a play of light and shadow.",
            "A panoramic view of mountain ranges where each peak seems to be hand-carved. The blue sky and snow-capped peaks create an impression of incredible beauty and harmony.",
            
            # For lakes
            "A photorealistic lake, its calm surface reflecting the surrounding mountains. The water seems so transparent that you can see the bottom at depth.",
            "A lake surrounded by wooded shores, where each tree is reflected in the water, creating a magical picture. Birds soar above the water surface, as if swimming in the air.",
            "Mountain lake surrounded by rocky cliffs, like a mirror reflecting the might of the mountains. The water has a deep emerald hue, like a treasure of nature.",
            
            # For mountain rivers
            "A mountain river, rushing tumultuously among rocky cliffs, creating an impression of the strength and power of nature. White water whirls play in the sunlight, like diamonds on the water surface.",
            "A natural mountain river running amidst green meadows and forests, like a living artery nourishing the surrounding nature with its waters. Deep canyons create a spectacular sight."
            
        ]
    }

    args = parser.parse_args()
    prompt = prompts[args.prompt]

    api = Text2ImageAPI('https://api-key.fusionbrain.ai/', 
                        '3AA09F0355ADE0B8AA21D98AA6793672', 
                        '1427DFB8BF8201240950DDCC76DED7F9')
    
    
    model_id = api.get_model()
    
    for i in tqdm(range(args.countImage)):
        try:
            uuid = api.generate(prompt[i % len(prompt)], model_id)
            images = api.check_generation(uuid)
            if images is not None:
                image_data = base64.b64decode(images[0])
                filename = f"{args.path}/A{i % len(prompt)}_{args.prompt.split('.')[0]} _ {r(0, 100000)}.jpg"
                while os.path.isfile(filename):
                    filename = f"{args.path}/A{+i % len(prompt)}_{args.prompt.split('.')[0]} _ {r(0, 100000)}.jpg"
                with open(filename, "wb+") as file:
                    file.write(image_data)
        except Exception as e:
            print(f"Error during image generation: {e}")
            print("Retrying to generate image...")
            try:
                model_id = api.get_model()  # Обновляем модель
            except Exception as e:
                print(f"Error getting model: {e}")
                print("Retrying to get the model...")
