import os
import argparse

from huggingface_hub import hf_hub_download

REPO_ID = "Laiyf/MieDB-100k"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args

def download(args):
    file_lists = []
    if args.train:
        print("Train Split Coming Soon")
        exit(0)
    else:
        print("Downloading MieDB-100k Benchmark Split")
        file_lists = ['dataBenchmark_00.tar']
    
    for file_name in file_lists:
        print(f"Downloading {file_name}...")
        hf_hub_download(repo_id=REPO_ID,
                        filename=file_name,
                        local_dir="./",
                        repo_type="dataset")


if __name__ == "__main__":
    args = parse_args()
    download(args)