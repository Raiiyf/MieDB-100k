import os
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
import os
import json
import tqdm

SEED = 42
MODEL_PATH = "stabilityai/sdxl-turbo"
BENCH_DIR = "../dataBenchmark"
OUTPUT_DIR = "eval_results"
OUTPUT_IMAGE_BATCH = 3

def main():
    with open(os.path.join(BENCH_DIR, 'metadata.json'), 'r') as f:
        test_sheet = json.load(f)

    pipeline = AutoPipelineForImage2Image.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    print("pipeline loaded")
    pipeline.to('cuda')
    pipeline.set_progress_bar_config(disable=None)

    output_folder = os.path.join(OUTPUT_DIR, "sdxl-turbo")
    os.makedirs(output_folder, exist_ok=True)
    for sample in tqdm.tqdm(test_sheet): 
        input_image_path = os.path.join(BENCH_DIR, sample['input_image'])
        input_image = Image.open(input_image_path).convert("RGB")
        prompt = sample['prompt']
        inputs = {
            "image": [input_image],
            "prompt": prompt,
            "generator": torch.manual_seed(SEED),
            "strength": 0.5,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "num_images_per_prompt": OUTPUT_IMAGE_BATCH,
        }
        with torch.inference_mode():
            output_path = os.path.join(output_folder, f"{sample['output_image'].split('/')[-1]}")
            if os.path.exists(output_path.replace(".png", f"_{OUTPUT_IMAGE_BATCH - 1}.png")):
                continue

            output = pipeline(**inputs)

            for i in range(len(output.images)):
                output_image = output.images[i]
                output_image.save(output_path.replace(".png", f"_{i}.png"))
            print("image saved at", os.path.abspath(f"{sample['output_image'].split('/')[-1]}"))


if __name__ == "__main__":
    main()