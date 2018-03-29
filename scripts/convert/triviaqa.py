import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Read dataset
with open(args.input) as f:
    dataset = json.load(f)

# Iterate and write question-answer pairs
count = 0
with open(args.input) as fi, open(args.output, 'w') as fo:
    for item in dataset['Data']:
        question = item['Question']
        answer = item['Answer']['Aliases']
        count += 1
        fo.write(json.dumps({'question': question, 'answer': answer}))
        fo.write('\n')

print('generated', count, 'qa pairs')
