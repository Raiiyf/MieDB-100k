# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

use_lora=false
model_path="OmniGen2/OmniGen2"

cleanup() {
    echo "Cleaning background processes..."
    kill $(jobs -p) 2>/dev/null
}

trap cleanup EXIT

unset CUDA_VISIBLE_DEVICES

NUM_GPU=4
for ((i=0; i<NUM_GPU; i++))
do
    # Calculate arithmetic values
    BEGIN=$i

    echo "Launching GPU $i: Processing from $BEGIN at stride $NUM_GPU"

    # Execute the command
    CUDA_VISIBLE_DEVICES=$i python inference_Miedb.py \
        --model_path "$model_path" \
        --num_inference_step 50 \
        --text_guidance_scale 5.0 \
        --image_guidance_scale 2.0 \
        --num_images_per_prompt 3 \
        --transformer_path experiments/ft_miedb/checkpoint-20000/transformer \
        --begin $BEGIN \
        --stride $NUM_GPU &
done

jobs -p

wait