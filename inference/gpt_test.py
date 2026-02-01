import json
import argparse
import requests
import os
import tqdm
import base64

URL = "https://api.openai.com/v1/images/edits"
api_key = ""
HEADERS = {
    "Authorization": f"Bearer {api_key}"
}

BENCH_DIR = "../dataBenchmark"
IMAGE_SAVE_FOLDER = "eval_results/gpt_image"
LOG_SAVE_FOLDER = "api_log/gpt_log"
os.makedirs(IMAGE_SAVE_FOLDER, exist_ok=True)
os.makedirs(LOG_SAVE_FOLDER, exist_ok=True)


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

        # Non-file fields are passed as a dictionary
        data = {
            "model": "gpt-image-1-0415-global",
            "prompt": entry["prompt"],
            "size": "1024x1024",
            "quality": "low",
            "output_compression": "100",
            "output_format": "png",
            "n": "1"
        }

        with open(os.path.join(BENCH_DIR, entry["input_image"]), "rb") as img0:
            files = {
                "image[0]": ("generated_image.png", img0, "image/png"),
            }
            response = requests.post(URL, headers=HEADERS, data=data, files=files).json()
            # print(response.json())

        try:
            result_dict = {
                "save_image": img_save_path,
                "usage": response["usage"],
            }

            with open(img_save_path, "wb") as f:
                f.write(base64.b64decode(response['data'][0]['b64_json']))

            with open(os.path.join(LOG_SAVE_FOLDER, f'{args.begin}_{args.stride}.jsonl'), 'a') as f:
                f.write(json.dumps(result_dict) + '\n')

        except:
            print(response)
