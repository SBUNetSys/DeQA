import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('answer_file', type=str)
    parser.add_argument('-s', '--save_file', type=str, default='out.txt')

    args = parser.parse_args()
    answer_file = args.answer_file

    with open(args.save_file, 'w') as f:
        for data_line in open(answer_file):
            data = json.loads(data_line)
            question = data['question']
            f.write(question + '\n')
