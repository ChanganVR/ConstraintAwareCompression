function results = bayesian_optimization(n_iter, init_points, input_caffemodel, last_constraint, ...
                                         current_constraint, output_prefix, original_latency, constraint_type)
    [~, ~, isloaded] = pyversion;
    if ~isloaded
        pyversion /local-scratch/changan-home/.pyenv/versions/py2/bin/python
    end
    py_path = '/local-scratch/changan-home/SkimCaffe/pruning/local-scratch/changan-home/SkimCaffe';
    sys_path = py.sys.path;
    if count(sys_path, py_path) == 0
        sys_path.append(py_path);
    end
    mod = py.importlib.import_module('pruning');
    py.reload(mod);
    
    conv1 = optimizableVariable('conv1', [0, 1]);
    conv2 = optimizableVariable('conv2', [0, 1]);
    conv3 = optimizableVariable('conv3', [0, 1]);
    conv4 = optimizableVariable('conv4', [0, 1]);
    conv5 = optimizableVariable('conv5', [0, 1]);
    fc6 = optimizableVariable('fc6', [0, 1]);
    fc7 = optimizableVariable('fc7', [0, 1]);
    fc8 = optimizableVariable('fc8', [0, 1]);
    
    
%     T = table(0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
%     T.Properties.VariableNames = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', ...
%                               'fc6', 'fc7', 'fc8'};
%     kwa = pyargs('input_caffemodel', input_caffemodel, 'current_constraint', current_constraint, ...
%                  'output_prefix', output_prefix);

    fun = @(input_params)alexnet_objective_function(input_params, input_caffemodel, last_constraint, ...
                                                    current_constraint, output_prefix, original_latency, constraint_type);
    
    results = bayesopt(fun, [conv1, conv2, conv3, conv4, conv5, fc6, fc7, fc8], ...
        'NumCoupledConstraints', 1, 'ExplorationRatio', 0.5, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', 'Verbose', 1, ...
        'MaxObjectiveEvaluations', n_iter, 'NumSeedPoints', init_points);
    results = 1;
end


function [objective, constraint] = alexnet_objective_function(P, input_caffemodel, last_constraint, ...
                                                              current_constraint, output_prefix, original_latency, ...
                                                              constraint_type)
    % wrapper for python alexnet objective function
    objective_func = py.pruning.objective_functions.matlab_alexnet_objective_function(...
        input_caffemodel, last_constraint, current_constraint, output_prefix, original_latency, constraint_type);
    
    kwa = pyargs('conv1', P.conv1, 'conv2', P.conv2, 'conv3', P.conv3, 'conv4', P.conv4, ...
        'conv5', P.conv5, 'fc6', P.fc6, 'fc7', P.fc7, 'fc8', P.fc8);
                       
    results = objective_func(kwa);
    objective = results{1};
    constraint = results{2};
end 