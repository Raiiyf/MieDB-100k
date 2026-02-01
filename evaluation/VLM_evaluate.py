from openai import OpenAI
import base64
import json
import tqdm
import os

llm_type = "gpt-5.2-1211-global"

client = OpenAI(
    api_key="",
    base_url="https://api.openai.com/v1",
)

model_name = "gemini_image"
mod_collage_folder = f"mod_collage/{model_name}"
batch_idx = 0

save_folder = "llm_score_result"
os.makedirs(save_folder, exist_ok=True)
result_file_path = f"{save_folder}/{model_name}.jsonl"
processed = set()
if os.path.exists(result_file_path):
    with open(result_file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data["image"])
            except (json.JSONDecodeError, KeyError):
                continue

metadata_path = "../dataBenchmark/metadata.json"
with open(metadata_path, 'r') as f:
    data_sheet = json.load(f)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

with open("rubric.txt", 'r') as f:
    prompt_system = f.read()

for entry in tqdm.tqdm(data_sheet):
    if entry["category"] != "Modification":
        continue
    try:
        collage_path = os.path.join(mod_collage_folder, entry['output_image'].split('/')[-1].replace('.png', f'_{batch_idx}.png'))
        if collage_path in processed:
            continue
        if not os.path.exists(collage_path):
            print(f"Missing sample {collage_path}!")
            continue

        response = client.chat.completions.create(
                        model=llm_type,
                        messages=[
                        {   
                            "role": "system",
                            "content": prompt_system
                        },
                        {   
                            "role": "user",
                            "content": [
                            {
                                "type": "text", "text": f"Instruction: {entry['prompt']}"
                            },
                            {
                                "type": "image_url",
                                "image_url": 
                                {
                                    "url": f"data:image/png;base64,{encode_image(collage_path)}", 
                                    "detail": "auto"
                                }
                            }]
                        }
                        ],
                        temperature=0,
                        max_tokens=1024,
                    )

        content = response.choices[0].message.content
        usage = response.usage
        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)

        result_data = {
            "image": collage_path,
            "score_result": content,
            "llm_type": llm_type,
            "usage": {
                "total_tokens": usage.total_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": cached_tokens
            }
        }

        with open(result_file_path, "a") as f:
            f.write(json.dumps(result_data) + "\n")

    except Exception as e:
        print(f"Error processing {entry['input_image']}: {e}")
