import sys

from src.knn.KNN_trainer import KNN_trainer
from src.rf.RF_trainer import RF_trainer
from src.svm.SVM_pred import SVM_pred
from src.svm.SVM_trainer import svm_trainer
from src.svm.validate import SVM_validate

if __name__ == '__main__':
    svm_trainer()

    # SVM_pred()
    #
    # RF_trainer()
    #
    # KNN_trainer()
    # SVM_validate()
