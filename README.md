# Constraint-Aware Deep Neural Network Compression
Given a real-time operational constraint(latency), this library will
automatically prune the network to satisfy the constraint while
preserving the accuracy. This library is based on [SkimCaffe](https://github.com/IntelLabs/SkimCaffe),
which has implemented direct sparse convolution operations and has an effective
speedup given an input sparse network. This framework can be applied to
different constraint types(latency, memory size), different network(Alexnet,
Resnet, Googlenet) and different datasets(imagenet, dtd).

For more technical details, please refer to the paper [Coming soon]().

## Usage
1. First follow the build-up instructions in [SkimCaffe](https://github.com/IntelLabs/SkimCaffe)
to build SkimCaffe.
2. Modify the cfp.config file to adapt to your own need(constraint values).
3.
3. Run training in root directory:
```
python2 main.py
```
4. To visualize the training result, specify the output directory:
```
python2 pruning/visualize_cbo_results.py [OUTPUT_DIR]
```
