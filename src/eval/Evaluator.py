class Evaluator:
    def __init__(self):
        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.tn = 0

    def __str__(self):
        print('TP:', self.tp)
        print('FP:', self.fp)
        print('FN:', self.fn)
        print('TN:', self.tn)
        print('TPR:', self.tpr())
        print('RECALL:', self.recall())
        print('FPR:', self.fpr())
        print('PRECISION:', self.precision())
        print('F-MEASURE:', self.f_measure())

    def confuse_matrix(self, pre, act):
        if pre == 1 and act == 1:
            self.tp += 1
            return 'TP'
        elif pre == 1 and act == 0:
            self.fp += 1
            return 'FP'
        elif pre == 0 and act == 1:
            self.fn += 1
            return 'FN'
        elif pre == 0 and act == 0:
            self.tn += 1
            return 'TN'

    def tpr(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def recall(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def fpr(self):
        if self.fp + self.tn == 0:
            return 0
        return self.fp / (self.fp + self.tn)

    def precision(self):
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def f_measure(self):
        r = self.recall()
        p = self.precision()
        if r + p == 0:
            return 0
        return 2 * r * p / (r + p)

    def reset(self):
        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.tn = 0
