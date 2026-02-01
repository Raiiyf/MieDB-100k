import numpy as np
import cv2
import os
import json
import tqdm
import pandas as pd

COLOR = {
    'RED': np.array([0, 0, 255]),
    'GREEN': np.array([0, 255, 0]),
    'BLUE': np.array([255, 0, 0]),
}

MASK_THRESHOLD = 0.5

def extract_alpha_multi_color(B, A, eps=1e-6):
    """
    Recover per-pixel alpha (mask) from a blue-blended image.
j
    B: blended image, shape (H, W, 3), dtype=float32 in 0-255
    A: background image (unblended), same shape/dtype
    color: overlay color (default to blue), 3-vector, e.g., [0,0,255]
    eps: small number to avoid division by zero
    Returns:
      alpha: per-pixel alpha in [0,1], shape (H, W)
      mask: binary mask (0 or 255) with alpha > 0.5
    """
    B = B.astype(np.float32)
    A = A.astype(np.float32)
    mask = np.zeros((A.shape[0], A.shape[1]))

    for O in COLOR.values(): 
        O = O.astype(np.float32)

        # per-pixel vectors
        t = B - A                 # B - A, shape (H, W, 3)
        v = O - A                 # O - A, shape (H, W, 3) because O is broadcast across pixels

        # dot products per pixel
        num = np.sum(t * v, axis=2)            # (H, W)
        den = np.sum(v * v, axis=2) + eps      # (H, W)

        alpha = num / den
        alpha = np.clip(alpha, 0.0, 1.0)

        mask += (alpha > 0.5).astype(np.uint8) * 255

    return alpha, mask

def extract_alpha(B, A, color="BLUE", eps=1e-6):
    """
    Recover per-pixel alpha (mask) from a blue-blended image.
j
    B: blended image, shape (H, W, 3), dtype=float32 in 0-255
    A: background image (unblended), same shape/dtype
    color: overlay color (default to blue), 3-vector, e.g., [0,0,255]
    eps: small number to avoid division by zero
    Returns:
      alpha: per-pixel alpha in [0,1], shape (H, W)
      mask: binary mask (0 or 255) with alpha > 0.5
    """
    O = COLOR[color]

    B = B.astype(np.float32)
    A = A.astype(np.float32)
    O = O.astype(np.float32)

    # per-pixel vectors
    t = B - A                 # B - A, shape (H, W, 3)
    v = O - A                 # O - A, shape (H, W, 3) because O is broadcast across pixels

    # dot products per pixel
    num = np.sum(t * v, axis=2)            # (H, W)
    den = np.sum(v * v, axis=2) + eps      # (H, W)

    alpha = num / den
    alpha = np.clip(alpha, 0.0, 1.0)

    mask = (alpha > 0.5).astype(np.uint8) # * 255

    return alpha, mask

def dice_score_binary(y_true, y_pred, threshold=None, smooth=1e-6):
    """
    Compute Dice score for binary masks.

    y_true: array-like of shape (H, W, ...) with truth mask (0/1 or False/True)
    y_pred: array-like same shape as y_true (can be probabilities)
    threshold: if not None, apply this threshold to y_pred (e.g., 0.5)
    smooth: small constant to avoid division by zero
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred)
    if threshold is not None:
        y_pred = y_pred >= threshold
    else:
        y_pred = y_pred.astype(bool)

    y_true_f = y_true.ravel()
    y_pred_f = y_pred.ravel()

    intersection = np.sum(y_true_f & y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    

if __name__ == "__main__":

    MODEL_NAME = "FluxKontext"
    RESULT_FOLDER_PATH = "../inference/eval_results"
    NUM_BATCH = 3

    BENCH_DIR = "../dataBenchmark"
    with open(os.path.join(BENCH_DIR, 'metadata.json'), 'r') as f:
        data_sheet = json.load(f)

    total_first_score = 0
    total_best_score = 0
    num_result = 0
    new_data_sheet = []
    print(f"Evaluating {MODEL_NAME}...")
    for sample in tqdm.tqdm(data_sheet):
        if sample['category'].lower() != 'perception':
            continue

        gt_image_path = os.path.join(BENCH_DIR, sample['output_image'])
        gt_image = cv2.imread(gt_image_path)
        origin_image = cv2.imread(os.path.join(BENCH_DIR, sample['input_image']))

        batch_result = []
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

                # gt_name = origin_name.replace('.png', f'_{color}.png')
                if sample.get('note', None) == "Multiclasses":
                    _, mask = extract_alpha_multi_color(inp_image, origin_image)
                    _, gt_mask = extract_alpha_multi_color(gt_image, origin_image)

                else:
                    color = gt_image_path.split('_')[-1].replace('.png', '').upper()
                    
                    _, mask = extract_alpha(inp_image, origin_image, color)
                    _, gt_mask = extract_alpha(gt_image, origin_image, color)

                batch_result.append(dice_score_binary(gt_mask, mask))
        
        if len(batch_result) == 0:
            batch_result.append(0)
        sample_first_score = batch_result[0]
        sample_best_score = max(batch_result)
        total_first_score += sample_first_score
        total_best_score += sample_best_score
        num_result += 1
    
        sample[f"{MODEL_NAME}-DICE@1"] = sample_first_score
        sample[f"{MODEL_NAME}-DICE@{NUM_BATCH}"] = sample_best_score
        new_data_sheet.append(sample)

    total_first_score /= num_result 
    total_best_score /= num_result 
    print(f"{MODEL_NAME} pass@1: {total_first_score}")
    print(f"{MODEL_NAME} pass@{NUM_BATCH}: {total_best_score}")
