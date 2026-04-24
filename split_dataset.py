import json
import random
import argparse

def load_dataset(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

def split_dataset(input_path = "MusicBench.json", train_path = "MusicBench_train.json", val_path = "MusicBench_val.json", val_ratio=0.1, seed=42):
    # Load dataset
    data = load_dataset(input_path)

    print(f"Total samples: {len(data)}")

    # Shuffle (reproducible)
    random.seed(seed)
    random.shuffle(data)

    # Split
    val_size = int(len(data) * val_ratio)
    train_data = data[val_size:]
    val_data = data[:val_size]

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Save
    with open(train_path, "w") as f:
        json.dump(train_data, f)

    with open(val_path, "w") as f:
        json.dump(val_data, f)

    print("Done.")