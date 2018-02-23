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


def test_accuracy(iter_cnt, solver, test_iters, output_caffemodel):
    solver.test_nets[0].share_with(solver.net)
    accuracy = 0
    loss = 0
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
        loss += solver.test_nets[0].blobs['loss'].data
    accuracy /= test_iters
    loss /= test_iters
    logging.info('Test in iteration {}, accuracy: {:.3f}, loss: {:.2f}'.format(iter_cnt, accuracy, loss))
    solver.net.save(output_caffemodel)
    logging.info('Model saved in {}'.format(output_caffemodel))
    return accuracy


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
    # some fine-tuning parameters
    test_iters = 1000
    test_interval = 1000
    disp_interval = 100
    min_acc = float(min_acc)
    max_iter = int(max_iter)

    if log_file is None:
        log_file = 'results/finetuning.log'
    if os.path.exists(output_caffemodel):
        # output_file = open(log_file, 'a+')
        raise ValueError('Fine-tuned caffemodel already exists')
    else:
        output_file = open(log_file, 'w')
    sys.stdout = output_file
    sys.stderr = output_file
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info('{:<20} {}'.format('Test iterations:', test_iters))
    logging.info('{:<20} {}'.format('Test interval:', test_interval))
    logging.info('{:<20} {}'.format('Display interval:', disp_interval))
    logging.info('{:<20} {}'.format('Min accuracy:', min_acc))
    logging.info('{:<20} {}'.format('Max iterations:', max_iter))

    solver = caffe.get_solver(solver_file)
    if not os.path.exists(output_caffemodel):
        solver.net.copy_from(input_caffemodel)
        logging.info('Fine-tune caffemodel from {}'.format(input_caffemodel))
    else:
        solver.net.copy_from(output_caffemodel)
        logging.info('Resume fine-tuning from {}'.format(output_caffemodel))
    iter_cnt = 0
    while iter_cnt < max_iter:
        # test
        if iter_cnt % test_interval == 0:
            # switch net to test mode
            acc = test_accuracy(iter_cnt, solver, test_iters, output_caffemodel)
            if iter_cnt == 0:
                accuracy_before = acc
            if acc >= min_acc:
                break

        # fine-tune
        solver.step(1)
        loss = solver.net.blobs['loss'].data
        if iter_cnt % disp_interval == 0 or iter_cnt + 1 == max_iter:
            logging.info('Training iteration {}, loss: {:.2f}'.format(iter_cnt, loss))
        iter_cnt += 1

    # test final accuracy
    accuracy_after = test_accuracy(iter_cnt, solver, test_iters, output_caffemodel)
    logging.info('Accuracy before: {:.3f}'.format(accuracy_before))
    logging.info('Accuracy after: {:.3f}'.format(accuracy_after))
    logging.info('Total iterations: {}'.format(iter_cnt))
    logging.info('Fine-tuning ends')
    output_file.close()


if __name__ == '__main__':
    assert len(sys.argv) >= 6
    fine_tune(*sys.argv[1:])
