import argparse

import os
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util


def remove_weights(inference_graph):
    out_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = node_def_pb2.NodeDef()
        tensor_proto = input_node.attr["value"].tensor
        if tensor_proto.tensor_content:
            output_node.op = input_node.op
            output_node.name = input_node.name
            dtype = input_node.attr["dtype"]
            output_node.attr["dtype"].CopyFrom(dtype)
            np_array = tensor_util.MakeNdarray(tensor_proto)
            output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(s=str(np_array).encode()))
            how_many_converted += 1
        else:
            output_node.CopyFrom(input_node)
        out_graph_def.node.extend([output_node])

    out_graph_def.library.CopyFrom(inference_graph.library)
    print("set %d weights to 0." % how_many_converted)
    return out_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='data/rnn_reader.npz')

    args = parser.parse_args()
    model_file = args.model

    sess = tf.Session()
    meta_graph_file = "{}.meta".format(model_file)
    if not os.path.exists(meta_graph_file):
        print("meta graph def:{} must exist if not given inference graph def".format(meta_graph_file))
        exit(-1)
    saver = tf.train.import_meta_graph(meta_graph_file, clear_devices=True)
    saver.restore(sess, model_file)

    model = os.path.splitext(os.path.basename(model_file))[0]
    print("{} model loaded".format(model))

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    # for node in input_graph_def.node:
    #     print("node:{}, op:{}, input:{}".format(node.name, node.op, node.input))
    print("{} model state restored".format(model))
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    with tf.gfile.FastGFile('data/{}.pb.txt'.format(model), "w") as gf:
        gf.write(str(remove_weights(graph_def)))
    # saver = tf.train.Saver()
    # saver.save(sess, 'data/tf_reader.ckpt')
