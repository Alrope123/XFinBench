import json
import os
from pathlib import Path
from utils_math import process_results

result_dir = "results/XFinBench/finqa"

# List all .json files under result_dir
json_files = list(Path(result_dir).glob("*.json"))
for json_file in json_files:
    results = []
    with open(json_file, "r") as f:
        for line in f:
            results.append(json.loads(line))
    
    is_correct = []
    got_number = []
    for result in results:
        parsed_results = process_results(result["answer"], [result["model_response"]])
        # print([parsed_results['model_answer']])
        is_correct.append(parsed_results['exact_match_flex'])
        got_number.append(parsed_results['got_number'])
    accuracy = sum(is_correct) / len(is_correct)
    got_number_ratio = sum(got_number) / len(got_number)
    print(f"Accuracy for {json_file.name}: {accuracy:.4f}")
    print(f"\tGot number ratio for {json_file.name}: {got_number_ratio:.4f}")

