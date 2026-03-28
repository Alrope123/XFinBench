import json

data = []
with open("dataset/finqa_test.json", "r") as f:
    for line in f:
        data.append(json.loads(line))

bool_data = []
calcu_data = []

for item in data:
    item['query'] = item['query'].split("\nAnswer:")[0]
    if item['answer'] in ['yes', 'no']:
        bool_data.append(item)
    else:
        try:
            float(item['answer'])
            calcu_data.append(item)
        except:
            print(f"Invalid answer: {item['answer']}")

with open("dataset/finqa_bool_test.json", "w") as f:
    for item in bool_data:
        json.dump(item, f)
        f.write("\n")

with open("dataset/finqa_calcu_test.json", "w") as f:
    for item in calcu_data:
        json.dump(item, f)
        f.write("\n")