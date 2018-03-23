import argparse

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

tf.contrib.eager.enable_eager_execution()
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
                                         shape=None,
                                         initializer=self.weight_initializer)
        self._bias = self.add_variable('bias',
                                       shape=None,
                                       initializer=self.bias_initializer)
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
                fw_cell = MyLSTMCell(num_units=hidden_size, name='lstm',
                                     weight_initializer=fw_weights, bias_initializer=fw_bias)
                bw_cell = MyLSTMCell(num_units=hidden_size, name='lstm',
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


def decode_one(scores_s, scores_e, batch, answers):
    score_s = tf.gather(scores_s, batch)
    score_e = tf.gather(scores_e, batch)

    scores = tf.matmul(tf.expand_dims(score_s, 1), tf.expand_dims(score_e, 0))
    # Zero out negative length and over-length span scores
    scores = ztg(scores, 14)
    max_score_idx = tf.argmax(tf.reshape(scores, [-1]))
    dim = scores.get_shape().as_list()[0]
    s_idx = tf.cast(tf.div(max_score_idx, dim), dtype=tf.float32)
    e_idx = tf.cast(tf.mod(max_score_idx, dim), dtype=tf.float32)
    s = tf.reduce_max(scores)

    answers = tf.concat([answers, tf.reshape(tf.stack([s_idx, e_idx, s]), [1, 3])], axis=0)
    return scores_s, scores_e, tf.add(batch, 1), answers


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

    def seq_attn_match(self, para_emb, q_emb, emb_dim):
        seq_weights = tf.get_variable('weights', initializer=self.qemb_match_weights)
        b = tf.get_variable('bias', initializer=self.qemb_match_bias)
        # Project vectors
        with tf.variable_scope('project_x'):
            x_re = tf.reshape(para_emb, [-1, emb_dim])  # 434 * 300
            x_pj = tf.matmul(x_re, seq_weights, transpose_b=True) + b  # 434 * 300
            x_pj = tf.nn.relu(x_pj)
            x_pj = tf.reshape(x_pj, [-1, tf.shape(para_emb)[1], emb_dim])  # 2 * 217 * 300

        with tf.variable_scope('project_y'):
            y_re = tf.reshape(q_emb, [-1, emb_dim])  # 12 * 300
            y_pj = tf.matmul(y_re, seq_weights, transpose_b=True) + b  # 12 * 300
            y_pj = tf.nn.relu(y_pj)
            y_pj = tf.reshape(y_pj, [-1, tf.shape(q_emb)[1], emb_dim])  # 2 * 6 * 300

        with tf.variable_scope('compute_scores'):
            # Compute scores
            scores = tf.matmul(x_pj, y_pj, transpose_b=True)  # 2 * 217 * 6

        with tf.variable_scope('normalize'):
            # Normalize with softmax
            alpha_flat = tf.reshape(scores, [-1, tf.shape(q_emb)[1]])  # 434 * 6
            alpha_flat = tf.nn.softmax(alpha_flat)

        with tf.variable_scope('weighted'):
            # Take weighted average
            alpha = tf.reshape(alpha_flat, [-1, tf.shape(para_emb)[1], tf.shape(q_emb)[1]])  # 2 * 217 * 6
            weighted_average = tf.matmul(alpha, q_emb)  # 2 * 217 * 300
        return weighted_average

    def linear_seq_attn(self, q_hidden):
        x_weight = tf.get_variable('weights', initializer=self.self_attn_weights)
        x_bias = tf.get_variable('bias', initializer=self.self_attn_bias)

        with tf.variable_scope('matmul'):
            x_flat = tf.reshape(q_hidden, [-1, tf.shape(q_hidden)[2]])
            scores = tf.reshape(tf.matmul(x_flat, x_weight, transpose_b=True) + x_bias, [-1, tf.shape(q_hidden)[1]])

        with tf.variable_scope('score'):
            scores = tf.exp(scores)
            x_sum = tf.expand_dims(tf.reduce_sum(scores, axis=1), axis=1)

        with tf.variable_scope('weighted'):
            scores = tf.expand_dims(tf.divide(scores, x_sum), axis=1)
            out = tf.squeeze(tf.matmul(scores, q_hidden), axis=1)

        return out

    def network(self, para_emb, para_feature, para_mask, q_emb):
        with tf.variable_scope('q_seq_attn'):
            q_weighted_emb = self.seq_attn_match(para_emb, q_emb, self.emb_shape[1])

        with tf.variable_scope('p_rnn_input'):
            para_rnn_input_list = [para_emb, q_weighted_emb, para_feature]
            para_rnn_input = tf.concat(para_rnn_input_list, axis=2)

        q_rnn_scope = 'question_rnn'
        q_rnn_weights = {k: v for k, v in self.weights.items() if k.startswith(q_rnn_scope)}
        question_hidden = stack_bi_rnn(input_data=q_emb,
                                       hidden_size=self.hidden_size,
                                       num_layers=self.question_layers,
                                       weights=q_rnn_weights,
                                       scope=q_rnn_scope)

        with tf.variable_scope('q_self_attn'):
            q_weighted_hidden = self.linear_seq_attn(question_hidden)

        para_rnn_scope = 'doc_rnn'
        para_rnn_weights = {k: v for k, v in self.weights.items() if k.startswith(para_rnn_scope)}
        para_hidden = stack_bi_rnn(input_data=para_rnn_input,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.doc_layers,
                                   weights=para_rnn_weights,
                                   scope=para_rnn_scope,
                                   mask=para_mask)

        with tf.variable_scope('start'):
            start_scores = bi_linear_seq_attn(self.start_attn_weights.transpose(), self.start_attn_bias,
                                              para_hidden, q_weighted_hidden, para_mask)

        with tf.variable_scope('end'):
            end_scores = bi_linear_seq_attn(self.end_attn_weights.transpose(), self.end_attn_bias,
                                            para_hidden, q_weighted_hidden, para_mask)

        with tf.variable_scope('answer'):
            final_answer = tf.concat([start_scores, end_scores], 1, name='scores')

        return final_answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_file', type=str, default='data/rnn_reader.npz')
    parser.add_argument('-e', '--embedding_file', type=str, default='data/emb.npz')
    parser.add_argument('-a', '--args', type=str, default='data/args.npy')
    parser.add_argument('-t', '--test_ex', type=str, default='data/ex.npz')

    args = parser.parse_args()
    ex_input = np.load(args.test_ex)
    ex_inputs = [ex_input[k] for k in ex_input.keys()[:-1]]
    emb = np.load(args.embedding_file)['emb']
    ex_inputs[0] = np.array([emb[i] for i in ex_inputs[0]])
    ex_inputs[3] = np.array([emb[i] for i in ex_inputs[3]])

    # np.savez_compressed('data/exn', **{str(k): v for k, v in enumerate(ex_inputs)})
    reader = RnnReader(args.args, args.weights_file)
    results = reader.network(*ex_inputs)
    print(results)
    answer = decode(results)
    print(answer)
