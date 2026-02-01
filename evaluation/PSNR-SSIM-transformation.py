import numpy as np
import cv2 
import os
import json
import tqdm
import pandas as pd

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

if __name__ == "__main__":

    MODEL_NAME = ""
    RESULT_FOLDER_PATH = "/mnt/workspace/workgroup/laiyongfan.lyf/Code/MedImageEdit/inference/eval_results"
    NUM_BATCH = 3

    BENCH_DIR = "../dataBenchmark"
    with open(os.path.join(BENCH_DIR, 'metadata.json'), 'r') as f:
        data_sheet = json.load(f)

    num_result = 0
    total_first_score = {
        'psnr': 0,
        'ssim': 0
    }
    total_best_score = {
        'psnr': 0,
        'ssim': 0
    }
    new_data_sheet = []
    print(f"Evaluating {MODEL_NAME}...")
    for sample in tqdm.tqdm(data_sheet):

        if sample['category'].lower() != 'transformation':
            continue

        gt_image_path = os.path.join(BENCH_DIR, sample['output_image'])
        gt_image = cv2.imread(gt_image_path)
        origin_image = cv2.imread(os.path.join(BENCH_DIR, sample['input_image']))

        batch_result = {
            "psnr": [],
            "ssim": []
        }
        for batch in range(NUM_BATCH):
            edit_image_path = os.path.join(RESULT_FOLDER_PATH, MODEL_NAME, gt_image_path.split('/')[-1])
            missing_flag = 0
            if not os.path.exists(edit_image_path.replace('.png', f'_{batch}.png')):
                print(f"Missing Sample {edit_image_path.replace('.png', f'_{batch}.png')}")
                inp_image = None
            else:
                inp_image = cv2.imread(edit_image_path.replace('.png', f'_{batch}.png'))

            if inp_image is None:
                print(f"Missing Sample {edit_image_path.replace('.png', f'_{batch}.png')}")
                missing_flag = 1

            if missing_flag != 1:
                if inp_image.shape != origin_image.shape:
                    inp_image = cv2.resize(inp_image, (origin_image.shape[1], origin_image.shape[0]))

                batch_result['psnr'].append(psnr(gt_image, inp_image))
                batch_result['ssim'].append(ssim(gt_image, inp_image, channel_axis=-1))
        
        if len(batch_result['psnr']) == 0:
            batch_result['psnr'].append(0)
            batch_result['ssim'].append(0)

        sample_first_score_psnr = batch_result['psnr'][0]
        sample_best_score_psnr = max(batch_result['psnr'])
        total_first_score['psnr'] += sample_first_score_psnr
        total_best_score['psnr'] += sample_best_score_psnr
        sample[f"{MODEL_NAME}-PSNR@1"] = sample_first_score_psnr
        sample[f"{MODEL_NAME}-PSNR@{NUM_BATCH}"] = sample_best_score_psnr

        sample_first_score_ssim = batch_result['ssim'][0]
        sample_best_score_ssim = max(batch_result['ssim'])
        total_first_score['ssim'] += sample_first_score_ssim
        total_best_score['ssim'] += sample_best_score_ssim
        sample[f"{MODEL_NAME}-SSIM@1"] = sample_first_score_ssim
        sample[f"{MODEL_NAME}-SSIM@{NUM_BATCH}"] = sample_best_score_ssim
        num_result += 1
    
        new_data_sheet.append(sample)

    total_first_score['psnr'] /= num_result 
    total_first_score['ssim'] /= num_result 
    total_best_score['psnr'] /= num_result 
    total_best_score['ssim'] /= num_result 
    print(f"{MODEL_NAME}-pass@1: {total_first_score}")
    print(f"{MODEL_NAME}-pass@{NUM_BATCH}: {total_best_score}")
