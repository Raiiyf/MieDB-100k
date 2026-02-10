import json
import argparse
import requests
import os
import tqdm
import base64

URL = "your/api/url/to/gemini-3-pro-image-preview"
api_key = ""

HEADERS = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

BENCH_DIR = "../dataBenchmark"
IMAGE_SAVE_FOLDER = "eval_results/gemini_image"
LOG_SAVE_FOLDER = "api_log/gemini_log"
os.makedirs(IMAGE_SAVE_FOLDER, exist_ok=True)
os.makedirs(LOG_SAVE_FOLDER, exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        # 1. Read the binary data
        # 2. Encode to base64 bytes
        # 3. Decode bytes to a UTF-8 string
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--begin",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    with open(os.path.join(BENCH_DIR, 'metadata.json'), 'r') as f:
        test_sheet = json.load(f)
    
    args = parse_args()
    test_sheet = test_sheet[args.begin::args.stride]

    batch_idx = 0
    for entry in tqdm.tqdm(test_sheet):
        img_save_path = os.path.join(IMAGE_SAVE_FOLDER, entry['output_image'].split('/')[-1]).replace('.png', f'_{batch_idx}.png')
        if os.path.exists(img_save_path):
            continue

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": entry["prompt"],
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": encode_image(os.path.join(BENCH_DIR, entry["input_image"]))
                            },
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": [
                    "IMAGE"
                ]
            }
        }

        response = requests.post(URL, headers=HEADERS, json=payload).json()

        try:
            result_dict = {
                "save_image": img_save_path,
                "usage": response["usage_metadata"],
                "model_version": response["modelVersion"]
            }

            with open(img_save_path, "wb") as f:
                f.write(base64.b64decode(response['candidates'][0]['content']['parts'][0]['inlineData']['data']))

            with open(os.path.join(LOG_SAVE_FOLDER, f'{args.begin}_{args.stride}.jsonl'), 'a') as f:
                f.write(json.dumps(result_dict) + '\n')

        except:
            print(response)
