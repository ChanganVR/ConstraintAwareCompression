from __future__ import print_function
from __future__ import division
import sys
import matplotlib.pyplot as plt


def plot_acc_iter(log_file):
    with open(log_file) as fo:
        lines = [line.strip().split() for line in fo.readlines()]
    accs = []
    iters = []
    for line in lines:
        iters.append(int(line[2].strip(',')))
        accs.append(float(line[-1]))

    plt.plot(iters, accs)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs iterations')
    plt.show()


if __name__ == '__main__':
    plot_acc_iter(sys.argv[1])
