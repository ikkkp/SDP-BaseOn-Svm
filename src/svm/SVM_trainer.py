import os
import pickle

import arff
import numpy as np
import yaml
from sklearn import svm

from src.eval.Evaluator import Evaluator


# bugs here
def standardization(data):
    return data
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def svm_trainer():
    config = yaml.load(open('./src/svm/config.yml', 'r'), Loader=yaml.FullLoader)
    # configurations000
    data_path = config['data_path']
    train_percent = config['train_percent']
    verbose = config['verbose']
    skip_validation = config['skip_validation']

    # load all datasets
    files = []
    for file in os.listdir(data_path):
        f = os.path.join(data_path, file)
        files.append(f)

    evaluator = Evaluator()

    # prepare data
    for file in files:
        print('*** ', file, ' ***')
        fp = open(file)
        testdata = arff.load(fp)
        a = list(testdata.get("data"))
        array = np.array(a)
        # array = np.array(list(arff.load(file)))
        # replace Y with 1 and N with 0
        for i in range(0, array.shape[0]):
            if array[i][-1] == 'Y':
                array[i][-1] = 1
            else:
                array[i][-1] = 0
        array = array.astype(float)
        size = array.shape
        print('        ', size, '\n')

        # divide data into train and test
        train_size = int(train_percent * size[0])
        x = array[:train_size, :-2]

        # standardize data
        for i in range(0, x.shape[0]):
            x[i, :] = standardization(x[i, :])
        y = array[:train_size, -1]

        # train with SVM
        print('=== 尝试训练 ===')
        clf = svm.SVC(kernel='linear', verbose=verbose)
        clf.fit(x, y)

        correct = 0
        total = 0

        if not skip_validation:
            print('\n=== 模型验证 ===')
            for r in range(train_size, size[0]):
                x = array[r][:-2]  # 38 dimensions
                print(x)
                y = array[r][-1]  # 1 dimension (1 or 0)
                if verbose:
                    print('输入：', x)
                    print('预测：', clf.predict([x])[0])
                    print('实际：', y)
                res = clf.predict([x])
                evaluator.confuse_matrix(res[0], y)
                if res[0] == y:
                    correct += 1
                    if verbose:
                        print('结果：正确', '\n')
                else:
                    if verbose:
                        print('结果：错误', '\n')
                total += 1
            print('=== 评价指标 ===\n')
            evaluator.__str__()
            print('\n===============')

        print('\n=== 导出模型 ===')
        # save model
        # 取最后一个文件名作为模型名
        model_name = './models/' + file.split('/')[-1] + '.model'
        with open(model_name, 'wb') as f:
            pickle.dump(clf, f)
        print(model_name)
        print('\n*********************************' + '\n\n\n')

        # reset evaluator
        evaluator.reset()
