from __future__ import print_function
from __future__ import division
import sys
import os
import logging
os.environ['GLOG_minloglevel'] = '0'
sys.path.append('/local-scratch/changan-home/SkimCaffe/python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()


def fine_tune(input_caffemodel, solver_file, output_caffemodel, min_acc, max_iter, log_file=None):
    """
    fine-tune until one of two requirements are satisfied: acc > min_acc or iter > max_iter
    :param input_caffemodel:
    :param solver_file:
    :param output_caffemodel:
    :param min_acc:
    :param max_iter:
    :return:
    """
    if log_file is None:
        log_file = 'results/finetuning.log'
    if os.path.exists(output_caffemodel):
        output_file = open(log_file, 'a+')
    else:
        output_file = open(log_file, 'w')
    sys.stdout = output_file
    sys.stderr = output_file
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    # some fine-tuning parameters
    test_iters = 1000
    test_interval = 1000
    disp_interval = 100
    # if log_file is not None:
    #     logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    # else:
    #     logging.basicConfig(filename='results/finetuning.log', filemode='w', level=logging.INFO)
    min_acc = float(min_acc)
    max_iter = int(max_iter)

    # solver_config = caffe.SolverParameter()
    # solver_config.train_net = 'models/bvlc_reference_caffenet/train_val_ft.prototxt'
    # solver_config.base_lr = 0.001
    # solver_config.momentum = 0.9
    # solver_config.weight_decay = 0.00005
    # solver_config.lr_policy = 'step'
    # solver_config.gamma = 0.01
    # solver_config.step = 1000
    # solver_config.max_iter = 100000
    # solver_config.snapshot = 2000
    # solver_config.snapshot_prefix = output_caffemodel
    # solver_config.type = 'sgd'
    # solver_config.display = 1
    # solver_config.regularization_type = 'L1'

    solver = caffe.get_solver(solver_file)
    if not os.path.exists(output_caffemodel):
        solver.net.copy_from(input_caffemodel)
        logging.info('Fine-tune caffemodel from {}'.format(input_caffemodel))
    else:
        solver.net.copy_from(output_caffemodel)
        logging.info('Resume fine-tuning from {}'.format(output_caffemodel))
    iter = 0
    while iter < max_iter:
        # test
        if iter % test_interval == 0:
            # switch net to test mode
            solver.test_nets[0].share_with(solver.net)
            accuracy = 0
            loss = 0
            for i in range(test_iters):
                solver.test_nets[0].forward()
                accuracy += solver.test_nets[0].blobs['accuracy'].data
                loss += solver.test_nets[0].blobs['loss'].data
            accuracy /= test_iters
            loss /= test_iters
            logging.info('Test in iteration {}, accuracy: {:.3f}, loss: {:.2f}'.format(iter, accuracy, loss))
            if accuracy >= min_acc:
                break
            solver.net.save(output_caffemodel)
            logging.info('Model saved in {}'.format(output_caffemodel))

        # fine-tune
        solver.step(1)
        loss = solver.net.blobs['loss'].data
        if iter % disp_interval == 0 or iter + 1 == max_iter:
            logging.info('Training iteration {}, loss: {:.2f}'.format(iter, loss))
        iter += 1

    # save learned weights
    solver.net.save(output_caffemodel)
    logging.info('Model saved in {}'.format(output_caffemodel))
    logging.info('Fine-tuning ends')
    output_file.close()


if __name__ == '__main__':
    assert len(sys.argv) >= 6
    fine_tune(*sys.argv[1:])
