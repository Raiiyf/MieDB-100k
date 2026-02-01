import os
import json

# with open("data_configs/train/example/edit/jsonls/0.jsonl", 'r', encoding='utf-8') as f:
#     data = [json.loads(line) for line in f if line.strip()]
#     print(len(data))
#     print(data)

# input_images, output_image, instruction, task_type (edit)

if __name__ == "__main__":

    TRAINDATA_DIR = os.path.abspath("../dataTrain")
    SAVE_PATH = f"data_configs/train/miedb/edit/jsonls/0.jsonl"

    with open(os.path.join(TRAINDATA_DIR, 'metadata.json'), 'r', encoding='utf-8') as f:
        samples = json.load(f)
        print(len(samples))

    data_sheet = []
    for sample in samples:
        data_dict = {
            "input_images": [os.path.join(TRAINDATA_DIR, sample["input_image"])],
            "output_image": os.path.join(TRAINDATA_DIR, sample["output_image"]),
            "instruction": sample["prompt"],
            "task_type": "edit"
        }
        data_sheet.append(data_dict)

    print(data_sheet[0])

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        for obj in data_sheet:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print("done!")
        