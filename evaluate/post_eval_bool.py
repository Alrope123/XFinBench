import json
import os
from pathlib import Path
from utils_bool import process_results

result_dir = "results/XFinBench/bool"

# List all .json files under result_dir
json_files = list(Path(result_dir).glob("*.json"))
for json_file in json_files:
    results = []
    with open(json_file, "r") as f:
        for line in f:
            results.append(json.loads(line))
    
    is_correct = []
    for result in results:
        ground_truth = "true" if result["ground_truth"] == "1.0" else "false"
        parsed_results = process_results(ground_truth, [result["model_response"]])
        # print([parsed_results['model_answer']])
        is_correct.append(parsed_results['exact_match_flex'])
    accuracy = sum(is_correct) / len(is_correct)
    print(f"Accuracy for {json_file.name}: {accuracy:.4f}")

