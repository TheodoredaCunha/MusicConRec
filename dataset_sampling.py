import json
import random

def sample_json(input_path, output_path, sample_size, seed=42):
    # Reproducibility
    random.seed(seed)

    # Load dataset
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON to be a list of entries.")

    if sample_size > len(data):
        raise ValueError(f"Sample size ({sample_size}) > dataset size ({len(data)})")

    # Random sample
    sampled = random.sample(data, sample_size)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(sampled)} samples to {output_path}")


# Example usage
sample_json(
    input_path="data/MusicBench_val.json",
    output_path="data/MusicBench_val_100.json",
    sample_size=100
)