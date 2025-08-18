import json

if __name__ == '__main__':

    path = 'data/o16/o16_training.gz'

    with open(path, 'r') as f:
        data = json.load(f)

    n = len(data)

    for i in range(n):

        levels = data[i]['levels'][0]
        observable_sets = data[i]['observable_sets']

        if i != 0:
            continue

        pp_in_index = observable_sets[0]['pp_in_index']
        pp_out_index = observable_sets[0]['pp_out_index']
        points = observable_sets[0]['points'][0]
