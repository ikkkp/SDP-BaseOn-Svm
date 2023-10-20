import pickle

import arff
import numpy as np

from src.eval.Evaluator import Evaluator
from src.svm.SVM_trainer import standardization


def SVM_validate():
    # load model
    with open('./models/KC3.arff.rf.model', 'rb') as f:
        clf = pickle.load(f)
    # load test data
    file = "./data/MDP/D'/KC3.arff"
    array = np.array(list(arff.load(file)))
    # replace Y with 1 and N with 0
    for i in range(0, array.shape[0]):
        if array[i][-1] == 'Y':
            array[i][-1] = 1
        else:
            array[i][-1] = 0
    # set data type to float
    array = array.astype(float)

    size = array.shape
    total = 0
    correct = 0
    N = 0
    evaluator = Evaluator()
    for r in range(0, size[0]):
        x = array[r][:-2]
        x = standardization(x)
        y = array[r][-1]
        print('输入：', x)
        res = clf.predict([x])[0]
        print('预测：', res)
        print('实际：', y)
        evaluator.confuse_matrix(res, y)
        if res == 0:
            N += 1
        if res == y:
            print('结果：正确', '\n')
            correct += 1
        else:
            print('结果：错误', '\n')
        total += 1
    print('\n\n正确率：', correct / total)
    print('预测出N的总数量：', N)
    print('预测出Y的总数量：', total - N)

    print('=== 评价指标 ===\n')
    evaluator.__str__()
    print('\n===============')
