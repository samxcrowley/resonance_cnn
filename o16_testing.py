import json

with open('o16_training.json') as f:

    data = json.load(f)

    print(data.keys())

    print(data["levels"][0].keys())
    print(data["observable_sets"][0].keys())
