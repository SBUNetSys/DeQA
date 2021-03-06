import argparse

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

np.set_printoptions(suppress=True)


class MyLSTMCell(BasicLSTMCell):

    def __init__(self, num_units, forget_bias=0,
                 state_is_tuple=True, activation=None,
                 reuse=None, name=None, weight_initializer=None, bias_initializer=None):
        super(MyLSTMCell, self).__init__(num_units, forget_bias, state_is_tuple, activation, reuse, name)
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self._kernel = None
        self._bias = None

    def compute_output_shape(self, input_shape):
        super(MyLSTMCell, self).compute_output_shape(input_shape)

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError('Expected inputs.shape[-1] to be known, saw shape: %s'
                             % inputs_shape)
        self._kernel = self.add_variable('kernel',
                                         shape=self.weight_initializer.shape,
                                         initializer=tf.constant_initializer(self.weight_initializer))
        self._bias = self.add_variable('bias',
                                       shape=self.bias_initializer.shape,
                                       initializer=tf.constant_initializer(self.bias_initializer))
        self.built = True

    def call(self, inputs, state):

        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # TensorFlow implementation
        # i, j, f, o = array_ops.split(
        # value=gate_inputs, num_or_size_splits=4, axis=one)
        # PyTorch version
        # in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        # need to adjust PyTorch weights, switch i and f

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, f, j, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=1)

        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(f)),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state


def length(sequence):
    with tf.variable_scope('pad'):
        eq = tf.equal(sequence, 0)
        length_ = tf.cast(eq, tf.int32)
        pad_len = tf.reduce_sum(length_, 1)
    return pad_len


def ztg(data, k):
    n = data.get_shape().as_list()[0]
    mask_u = tf.greater_equal(tf.tile(tf.reshape(tf.range(n), [n, 1]), [1, n]),
                              tf.tile(tf.reshape(tf.range(1, n + 1), [1, n]), [n, 1]))
    mask_l = tf.greater_equal(tf.tile(tf.reshape(tf.range(n), [n, 1]), [1, n]),
                              tf.tile(tf.reshape(tf.range(-k + 1, n - k + 1), [1, n]), [n, 1]))
    tri = tf.where(mask_u, tf.zeros(data.get_shape(), dtype=data.dtype), data)  # tri_u
    # print(tri)
    tri2 = tf.where(mask_l, tri, tf.zeros(data.get_shape(), dtype=data.dtype))  # tri_l
    return tri2


def decode(answer_scores):
    answers = []
    for scores in answer_scores:
        score_s, score_e = tf.split(scores, 2)
        scores = tf.matmul(tf.expand_dims(score_s, 1), tf.expand_dims(score_e, 0))
        # Zero out negative length and over-length span scores
        scores = ztg(scores, 14)
        max_score_idx = tf.argmax(tf.reshape(scores, [-1]))
        dim = scores.get_shape().as_list()[0]
        s_idx = tf.cast(tf.div(max_score_idx, dim), dtype=tf.float32)
        e_idx = tf.cast(tf.mod(max_score_idx, dim), dtype=tf.float32)
        s = tf.reduce_max(scores)
        answers.append([s_idx, e_idx, s])
    return answers


def stack_bi_rnn(input_data, hidden_size, num_layers, weights, scope, mask=None):
    rnn_scope = 'q_rnn' if scope.startswith('q') else 'p_rnn'
    with tf.variable_scope(rnn_scope):
        if mask is not None:
            seq_len = length(mask)
        else:
            seq_len = tf.ones(tf.shape(input_data)[0], dtype=tf.int32) * tf.shape(input_data)[1]
        outputs = []
        last_output = input_data
        for k in range(num_layers):
            fw_weights_input = weights['{}.rnns.{}.weight_ih_l0'.format(scope, str(k))]
            fw_weights_hidden = weights['{}.rnns.{}.weight_hh_l0'.format(scope, str(k))]
            fw_weights = np.concatenate((fw_weights_input, fw_weights_hidden), axis=1).transpose()

            fw_bias_input = weights['{}.rnns.{}.bias_ih_l0'.format(scope, str(k))]
            fw_bias_hidden = weights['{}.rnns.{}.bias_hh_l0'.format(scope, str(k))]
            fw_bias = np.add(fw_bias_input, fw_bias_hidden)

            bw_weights_input = weights['{}.rnns.{}.weight_ih_l0_reverse'.format(scope, str(k))]
            bw_weights_hidden = weights['{}.rnns.{}.weight_hh_l0_reverse'.format(scope, str(k))]
            bw_weights = np.concatenate((bw_weights_input, bw_weights_hidden), axis=1).transpose()

            bw_bias_input = weights['{}.rnns.{}.bias_ih_l0_reverse'.format(scope, str(k))]
            bw_bias_hidden = weights['{}.rnns.{}.bias_hh_l0_reverse'.format(scope, str(k))]
            bw_bias = np.add(bw_bias_input, bw_bias_hidden)

            with tf.variable_scope('layer_{}'.format(str(k))):
                fw_cell = MyLSTMCell(num_units=hidden_size, name='basic_lstm_cell',
                                     weight_initializer=fw_weights, bias_initializer=fw_bias)
                bw_cell = MyLSTMCell(num_units=hidden_size, name='basic_lstm_cell',
                                     weight_initializer=bw_weights, bias_initializer=bw_bias)

                output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, last_output,
                                                            dtype=tf.float32,
                                                            sequence_length=seq_len,
                                                            scope='bi_rnn')
                last_output = tf.concat(output, 2)
            outputs.append(last_output)
        out = tf.concat(outputs, axis=2)
    return out


def bi_linear_seq_attn(bi_w, bi_bias, x, y, x_mask):
    with tf.variable_scope('bi_attn'):
        w = tf.get_variable('weights', initializer=bi_w)
        b = tf.get_variable('bias', initializer=bi_bias)

        with tf.variable_scope('weighted'):
            wy = tf.add(tf.matmul(y, w), b)
        with tf.variable_scope('linear'):
            xwy = tf.matmul(x, tf.expand_dims(wy, 2))
            xwy = tf.squeeze(xwy, 2, name='alpha')
        with tf.variable_scope('score'):
            alpha = tf.exp(xwy)
            z = tf.cast(tf.equal(x_mask, 0), dtype=tf.float32)
            out = tf.multiply(alpha, z)
    return out


class RnnReader(object):
    def __init__(self, arg, weights_file):
        # noinspection PyTypeChecker
        opt = dict(np.load(arg).item())
        self.args = opt
        self.emb_shape = opt['embedding_shape']
        self.hidden_size = opt['hidden_size']
        self.doc_layers = opt['doc_layers']
        self.question_layers = opt['question_layers']
        self.weights = np.load(weights_file)  # must end with .npz
        self.qemb_match_weights = self.weights['qemb_match.linear.weight']
        self.qemb_match_bias = self.weights['qemb_match.linear.bias']
        self.self_attn_weights = self.weights['self_attn.linear.weight']
        self.self_attn_bias = self.weights['self_attn.linear.bias']
        self.start_attn_weights = self.weights['start_attn.linear.weight']
        self.start_attn_bias = self.weights['start_attn.linear.bias']
        self.end_attn_weights = self.weights['end_attn.linear.weight']
        self.end_attn_bias = self.weights['end_attn.linear.bias']

        self.sess = tf.Session()

    def seq_attn_match(self, x, y, input_size):
        seq_weights = tf.get_variable('weights', initializer=self.qemb_match_weights)
        b = tf.get_variable('bias', initializer=self.qemb_match_bias)
        # Project vectors
        with tf.variable_scope('project_x'):
            x_re = tf.reshape(x, [-1, input_size])
            x_pj = tf.matmul(x_re, seq_weights, transpose_b=True) + b
            x_pj = tf.nn.relu(x_pj)
            x_pj = tf.reshape(x_pj, [-1, tf.shape(x)[1], input_size])

        with tf.variable_scope('project_y'):
            y_re = tf.reshape(y, [-1, input_size])
            y_pj = tf.matmul(y_re, seq_weights, transpose_b=True) + b
            y_pj = tf.nn.relu(y_pj)
            y_pj = tf.reshape(y_pj, [-1, tf.shape(y)[1], input_size])

        with tf.variable_scope('compute_scores'):
            # Compute scores
            scores = tf.matmul(x_pj, y_pj, transpose_b=True)

        with tf.variable_scope('normalize'):
            # Normalize with softmax
            alpha_flat = tf.reshape(scores, [-1, tf.shape(y)[1]])
            alpha_flat = tf.nn.softmax(alpha_flat)

        with tf.variable_scope('weighted'):
            # Take weighted average
            alpha = tf.reshape(alpha_flat, [-1, tf.shape(x)[1], tf.shape(y)[1]])
            weighted_average = tf.matmul(alpha, y)
        return weighted_average

    def linear_seq_attn(self, x):
        x_weight = tf.get_variable('weights', initializer=self.self_attn_weights)
        x_bias = tf.get_variable('bias', initializer=self.self_attn_bias)

        with tf.variable_scope('matmul'):
            x_flat = tf.reshape(x, [-1, tf.shape(x)[2]])
            scores = tf.reshape(tf.matmul(x_flat, x_weight, transpose_b=True) + x_bias, [-1, tf.shape(x)[1]])

        with tf.variable_scope('score'):
            scores = tf.exp(scores)
            x_sum = tf.expand_dims(tf.reduce_sum(scores, axis=1), axis=1)

        with tf.variable_scope('weighted'):
            scores = tf.expand_dims(tf.divide(scores, x_sum), axis=1)
            out = tf.squeeze(tf.matmul(scores, x), axis=1)

        return out

    def network(self, x1_emb, x1_mask, x2_emb):
        with tf.variable_scope('q_seq_attn'):
            x2_weighted_emb = self.seq_attn_match(x1_emb, x2_emb, self.emb_shape[1])

        with tf.variable_scope('p_rnn_input'):
            doc_rnn_input_list = [x1_emb, x2_weighted_emb]
            doc_rnn_input = tf.concat(doc_rnn_input_list, axis=2)

        # self.np_rnn(x2_emb.numpy())
        q_rnn_scope = 'question_rnn'
        q_rnn_weights = {k: v for k, v in self.weights.items() if k.startswith(q_rnn_scope)}
        question_hidden = stack_bi_rnn(input_data=x2_emb,
                                       hidden_size=self.hidden_size,
                                       num_layers=self.question_layers,
                                       weights=q_rnn_weights,
                                       scope=q_rnn_scope)

        with tf.variable_scope('q_self_attn'):
            q_weighted_hidden = self.linear_seq_attn(question_hidden)

        doc_rnn_scope = 'doc_rnn'
        doc_rnn_weights = {k: v for k, v in self.weights.items() if k.startswith(doc_rnn_scope)}
        doc_hidden = stack_bi_rnn(input_data=doc_rnn_input,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.doc_layers,
                                  weights=doc_rnn_weights,
                                  scope=doc_rnn_scope,
                                  mask=x1_mask)

        with tf.variable_scope('start'):
            start_scores = bi_linear_seq_attn(self.start_attn_weights.transpose(), self.start_attn_bias,
                                              doc_hidden, q_weighted_hidden, x1_mask)

        with tf.variable_scope('end'):
            end_scores = bi_linear_seq_attn(self.end_attn_weights.transpose(), self.end_attn_bias,
                                            doc_hidden, q_weighted_hidden, x1_mask)

        with tf.variable_scope('answer'):
            final_answer = tf.concat([start_scores, end_scores], 1, name='scores')

        # batches = start_scores.get_shape().as_list()[0]
        # idx = tf.constant(0)
        #
        # def cond(_s, _e, idx_, _a):
        #     return idx_ < (batches or 1)
        #
        # answers = tf.constant([-1, -1, -1.0], dtype=tf.float32, shape=[1, 3])
        # final_results = tf.while_loop(cond, decode_one, [start_scores, end_scores, idx, answers],
        #                               shape_invariants=[start_scores.get_shape(), end_scores.get_shape(),
        #                                                 idx.get_shape(), tf.TensorShape([None, 3])])
        # final_answer = tf.identity(final_results[-1], name='answer')
        return final_answer


def remove_weights(inference_graph):
    out_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = node_def_pb2.NodeDef()
        tensor_proto = input_node.attr['value'].tensor
        if tensor_proto.tensor_content:
            output_node.op = input_node.op
            output_node.name = input_node.name
            dtype = input_node.attr['dtype']
            output_node.attr['dtype'].CopyFrom(dtype)
            np_array = tensor_util.MakeNdarray(tensor_proto)
            output_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(s=str(np_array).encode()))
            how_many_converted += 1
        else:
            output_node.CopyFrom(input_node)
        out_graph_def.node.extend([output_node])

    out_graph_def.library.CopyFrom(inference_graph.library)
    print('set %d weights to 0.' % how_many_converted)
    return out_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_file', type=str, default='data/rnn_reader.npz')
    parser.add_argument('-e', '--embedding_file', type=str, default='data/emb.npz')
    parser.add_argument('-a', '--args', type=str, default='data/args.npy')
    parser.add_argument('-t', '--test_ex', type=str, default='data/ex.npz')
    parser.add_argument('-egr', '--eager', action='store_true')

    args = parser.parse_args()

    ex_input = np.load(args.test_ex)
    ex_inputs = [ex_input[k] for k in ex_input.keys()[:-1]]
    emb = np.load(args.embedding_file)['emb']
    ex_inputs[0] = np.array([emb[i] for i in ex_inputs[0]])
    ex_inputs[3] = np.array([emb[i] for i in ex_inputs[3]])

    inputs = [ex_inputs[0], ex_inputs[2], ex_inputs[3]]
    if args.eager:
        tf.enable_eager_execution()
        reader = RnnReader(args.args, args.weights_file)
        results = reader.network(*inputs)
        print(results)
        answer = decode(results)
        print(answer)
    else:
        reader = RnnReader(args.args, args.weights_file)
        input_names = ['para/emb', 'para/mask', 'q_emb']

        placeholders = [tf.placeholder(tf.as_dtype(k.dtype), shape=[None, None, *k.shape[2:]], name=n)
                        for n, k in zip(input_names, inputs)]
        final_answers = reader.network(*placeholders)
        sess = reader.sess
        np_weights = dict()
        for var in tf.global_variables():
            sess.run(var.initializer)
            # print(sess.run(var))
            var_name = var.name[:-2]
            print(var_name, var.shape)
            np_weights[var_name] = sess.run(var)
        np.savez_compressed('data/drqa_w', **{k: v for k, v in np_weights.items()})
        sess.run(tf.global_variables_initializer())
        # print(sess.run(tf.report_uninitialized_variables()))

        results = sess.run(final_answers, feed_dict={k: v for k, v in zip(placeholders, inputs)})
        print(results)
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names=['answer/scores']  # The output node names are used to select the useful nodes
        )
        with tf.gfile.GFile('data/drqa.pb', 'wb') as gf:
            gf.write(output_graph_def.SerializeToString())
        with tf.gfile.FastGFile('data/drqa.pb.txt', 'w') as gf:
            gf.write(str(remove_weights(graph_def)))
    # saver = tf.train.Saver()
    # saver.save(sess, 'data/tf_reader.ckpt')
