import torch
import os
import torch
from PIL import Image
from diffusers import Step1XEditPipelineV1P2
import os
import json
import tqdm
import argparse

SEED = 42
BENCH_DIR = "../dataBenchmark"
OUTPUT_DIR = "eval_results"
OUTPUT_IMAGE_BATCH = 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--begin_index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args

def main():
    with open(os.path.join(BENCH_DIR, 'metadata,json'), 'r') as f:
        test_sheet = json.load(f)
    args = parse_args()
    if args.end_index == 0:
        args.end_index = len(test_sheet)
    test_sheet = test_sheet[args.begin_index:args.end_index]

    pipeline = Step1XEditPipelineV1P2.from_pretrained("stepfun-ai/Step1X-Edit-v1p2", torch_dtype=torch.bfloat16)
    print("pipeline loaded")
    pipeline.to('cuda')
    pipeline.set_progress_bar_config(disable=None)

    output_folder = os.path.join(OUTPUT_DIR, "Step1X-Edit-v1p2")
    os.makedirs(output_folder, exist_ok=True)
    for sample in tqdm.tqdm(test_sheet): 
        input_image_path = os.path.join(BENCH_DIR, sample['input_image'])
        input_image = Image.open(input_image_path).convert("RGB")
        prompt = sample['prompt']
        inputs = {
            "image": input_image,
            "prompt": prompt,
            "generator": torch.manual_seed(SEED),
            "size_level": 512,
            "height": min(input_image.size[0], 512),
            "width": min(input_image.size[1], 512),
            "true_cfg_scale": 6,
            "num_inference_steps": 50,
            "num_images_per_prompt": OUTPUT_IMAGE_BATCH,
        }

        with torch.inference_mode():
            output_path = os.path.join(output_folder, f"{sample['output_image'].split('/')[-1]}")
            if os.path.exists(output_path.replace(".png", f"_0.png")):
                continue

            output = pipeline(**inputs)
            for i in range(len(output.images)):
                output_image = output.images[i]
                output_image.save(output_path.replace(".png", f"_{i}.png"))
            print("image saved at", os.path.abspath(f"{sample['output_image'].split('/')[-1]}"))

if __name__ == "__main__":
    main()