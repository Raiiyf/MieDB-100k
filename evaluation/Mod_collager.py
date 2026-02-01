# Generate collage image for VLM evaluation in Modification perspective

import cv2
import numpy as np
import os
import json
import tqdm

model_name = "gemini_image"
batch_idx = 0
sample_folder = f"../inference/eval_results/{model_name}"
BENCH_DIR = "../dataBenchmark"
collage_folder = f"mod_collage/{model_name}"
os.makedirs(collage_folder, exist_ok=True)

with open(os.path.join(BENCH_DIR, 'metadata.json'), 'r') as f:
    data_sheet = json.load(f)

print(f"generate collage for {model_name}")
for entry in tqdm.tqdm(data_sheet):
    if entry["category"] != "Modification":
        continue

    input_image = cv2.imread(os.path.join(BENCH_DIR, entry["input_image"]))
    model_result = cv2.imread(os.path.join(sample_folder, entry["input_image"].split('/')[-1]).replace('.png', f'_{batch_idx}.png'))
    output_image = cv2.imread(os.path.join(BENCH_DIR, entry["output_image"]))
    if model_result.shape != input_image.shape:
        model_result = cv2.resize(model_result, (input_image.shape[1], input_image.shape[0]))
    collage = np.concatenate([input_image, model_result, output_image], axis=1)

    cv2.imwrite(os.path.join(collage_folder, entry["input_image"].split('/')[-1]).replace('.png', f'_{batch_idx}.png'), collage)
