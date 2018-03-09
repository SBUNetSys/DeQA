import argparse
import time

import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--frozen_model', type=str, default='data/tf_reader.pb')
    parser.add_argument('-t', '--test_ex', type=str, default='data/ex.npz')
    parser.add_argument('-e', '--embedding_file', type=str, default='data/emb.npz')

    args = parser.parse_args()
    ex_input = np.load(args.test_ex)
    ex_inputs = [ex_input[k] for k in ex_input.keys()[:-1]]
    emb = np.load(args.embedding_file)['emb']
    ex_inputs[0] = np.array([emb[i] for i in ex_inputs[0]])
    ex_inputs[3] = np.array([emb[i] for i in ex_inputs[3]])

    frozen_model = args.frozen_model
    # reader.network(*(ex_input[k] for k in ex_input.keys()))
    with tf.gfile.GFile(frozen_model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                            name='', op_dict=None, producer_op_list=None)
    session = tf.Session(graph=graph)
    # weight = session.run(graph.get_operation_by_name('p_rnn/layer_0/bi_rnn/fw/lstm/kernel').outputs[0])
    # print(weight)
    begin_time = time.time()
    input_names = ['para/emb', 'para/feature', 'para/mask', 'q_emb']

    placeholders = [graph.get_operation_by_name(name).outputs[0] for name in input_names]

    output = graph.get_operation_by_name('answer/scores').outputs[0]

    scores = session.run(output, feed_dict={k: v for k, v in zip(placeholders, ex_inputs)})
    print(scores)
    end_time = time.time()
