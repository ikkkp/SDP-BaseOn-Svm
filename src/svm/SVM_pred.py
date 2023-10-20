import codecs
import os
import pickle

import arff
import numpy as np

from src.svm.SVM_trainer import standardization


def SVM_pred():
    # 从命令行读取文件
    # module = input('请输入要预测的模块路径：')
    # 测试集路径
    metrics_res = "K:\\"
    file = metrics_res + "test.arff"

    # print('*** ', file, ' ***')
    fp = open(file)
    testdata = arff.load(fp)
    a=list(testdata.get("data"))
    array = np.array(a)
    size = array.shape


    # print('        ', size, '\n')

    # print('=== 载入模型 ===')
    # 从文件中读取模型
    with open('./model2/KC3.arff.knn.model', 'rb') as f:
        clf = pickle.load(f)

    # print('=== 模型验证 ===')
    N = 0
    total = 0
    re = ""
    for r in range(0, size[0]):
        total += 1
        x = array[r][:-1]
        x = standardization(x)
        # print('模块 ' + r.__str__() + '：\n', x)
        res = clf.predict([x])[0]
        # print('预测结果：', res)
        if res == 0:
            re += 'N'
        else:
            re += 'Y'
        if r != size[0] - 1:
            re += ','
        if res == 0:
            N += 1
        # print('\n================================================================================\n\n')
    with open('./result/result.txt', 'w') as f:
        f.write(re.__str__())
    # print('预测出N的总数量：', N)
    # print('预测出Y的总数量：', total - N)
