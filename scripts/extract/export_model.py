#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', type=str, help='PyTorch model file')
    parser.add_argument('-o', '--out_file', type=str, help='numpy npz file for model weights, except emb')
    parser.add_argument('-e', '--embedding_file', type=str, help='embedding weights in numpy npz format')
    args = parser.parse_args()

    model_weights = torch.load(args.model_file, map_location='cpu')
    if args.embedding_file:
        emb_weights = model_weights['state_dict']['embedding.weight']
        print('saving emb weights to', args.embedding_file)
        np.savez_compressed(args.embedding_file, emb=emb_weights)
        print('emb weights saved')

    name_weight_dict = dict()
    for k, v in model_weights['state_dict'].items():
        if k == 'embedding.weight':
            # print('skip emb weights')
            continue
        else:
            print(k, v.shape)
            name_weight_dict[k] = v.numpy()
    np.savez_compressed(args.out_file, **name_weight_dict)
    print('model weights saved to', args.out_file)
    with open(args.out_file + '.txt', 'w', encoding='utf-8')as f:
        f.write('\n'.join(['{}, {}'.format(k, v.shape) for k, v in name_weight_dict.items()]))
