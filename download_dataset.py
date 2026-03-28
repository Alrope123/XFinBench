from datasets import load_dataset

dataset = load_dataset("ChanceFocus/flare-finqa", trust_remote_code=True)

dataset["test"].to_json(f"finqa_test.json")

print(f"Saved finqa_test.json")