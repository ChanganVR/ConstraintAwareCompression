from __future__ import print_function
import numpy as np
import sys
import os


def prune(template_caffemodel, input_caffemodel, output_caffemodel, prototxt_file):
    # logging.basicConfig(filename='results/prune_debug.log', filemode='w', level=logging.DEBUG)
    # surpress log output from caffe loading
    # os.environ['GLOG_minloglevel'] = '2'
    import caffe
    template_net = caffe.Net(prototxt_file, template_caffemodel, caffe.TEST)
    layer_dict = dict()
    for layer in template_net.params:
        layer_dict[layer] = template_net.params[layer][0].data == 0
    del template_net

    input_net = caffe.Net(prototxt_file, input_caffemodel, caffe.TEST)
    for layer in input_net.params:
        input_net.params[layer][0].data[layer_dict[layer]] = 0
    input_net.save(output_caffemodel)
    print('Pruning done.')


if __name__ == '__main__':
    prune(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
