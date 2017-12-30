#encoding=utf-8

import pandas as pd
import numpy as np
import math
import cv2
import random
import time
import pprint as pp

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SVM(object):
    """Summary of class here.

    Attributes:
        rate:       learning rate
        params of dual form svm: a and b;
        features:   training set, input vectors of features
        labels:     training set, labels of features;
    """

    def __init__(self, inputs, labels, soft_param=1, error_thres=0.01, study_max=10000, nochange_max=50):
        """Inits BinaryPerceptron."""
        self.inputs = inputs
        self.labels = labels
        self.n = self.inputs.shape[0]
        self.C = soft_param
        self.t = error_thres
        # a和b参见svm对偶形式中的参数
        self.a = np.zeros(self.n)
        self.b = 0
        # SMO算法中用来存放误差的数组
        # self.e = np.zeros(inputs.size)
        # for i in range(self.n):
        #     self.e[i] = self._error(self.inputs[i], self.labels[i])
        self.study_max = study_max
        self.nochange_max = nochange_max

    def train(self):
        """
        Using SMO method to learn a and b from training set.
        """
        counter = 0
        while True:
            if counter > self.study_max:
                break
            counter += 1
            (i, j) = self._select()
            if i == -1:
                break
            tmp = self.a[i] + self.a[j]
            L = max(0, tmp - self.C)
            H = min(self.C, tmp)
            if (self.labels[i] != self.labels[j]):
                tmp = self.a[j] - self.a[i]
                L = max(0, tmp)
                H = min(self.C, self.C + tmp)
            xi = self.inputs[i]
            yi = self.labels[i]
            xj = self.inputs[j]
            yj = self.labels[j]
            ei = self._error(xi, yi)
            ej = self._error(xj, yj)
            aj_new = self.a[j] + yj * (ei - ej) / \
                (self._kernel(xi, xi) + self._kernel(xj, xj) - 2 * self._kernel(xi, xj))
            print aj_new
            print H
            if aj_new > H:
                aj_new = H
            elif aj_new < L:
                aj_new = L
            ai_new = yi * yj * (self.a[j] - aj_new)
            
            bi_new = - ei - yi * self._kernel(xi, xi) * (ai_new - self.a[i]) \
                - yj * self._kernel(xi, xj) * (aj_new - self.a[j]) + self.b
            bj_new = - ej - yi * self._kernel(xi, xj) * (ai_new - self.a[i]) \
                - yj * self._kernel(xj, xj) * (aj_new - self.a[j]) + self.b
            if ai_new > 0 and ai_new < self.C and aj_new > 0 and aj_new < self.C:
                self.b = bi_new
            else:
                self.b = float(bi_new + bj_new) / 2
            self.a[i] = ai_new
            self.a[j] = aj_new
        pp.pprint(self.a)
        pp.pprint(self.b)


    def predict(self, x):
        y = self._output(x)
        if y >= 0: 
            return 1
        else:
            return -1

    def _select(self):
        (res1, res2) = (-1, -1)
        for i in range(self.n):
            if self.a[i] < self.C and self.a[i] > 0:
                if (math.fabs(self.labels[i] * self._output(self.inputs[i]) - 1) > self.t):
                    res1 = i
                    break
        if res1 == -1:
            for i in range(self.n):
                if self.a[i] == 0:
                    if (self.labels[i] * self._output(self.inputs[i]) <= 1 - self.t):
                        res1 = i
                        break
                elif self.a[i] == self.C:
                    if (self.labels[i] * self._output(self.inputs[i]) >= 1 + self.t):
                        res1 = i
                        break
        if res1 == -1:
            return (res1, res2)

        error1 = self._error(self.inputs[res1], self.labels[res1])
        maxE = -1
        res2 = -1
        for i in range(self.n):
            if i == res1:
                continue
            error2 = self._error(self.inputs[i], self.labels[i])
            ediff = math.fabs(error2 - error1)
            if ediff >  maxE:
                res2 = i
                maxE = ediff
        return (res1, res2)

    def _output(self, x):
        dotsum = 0
        for i in range(self.n):
            dotsum += self._kernel(x, self.inputs[i]) * self.labels[i] * self.a[i]
        return dotsum + self.b

    def _error(self, x, y):
        return self._output(x) - y;


    def _kernel(self, x1, x2, type='basic'):
        if type == 'basic':
            return np.dot(x1, x2)
        elif type == 'poly': 
            # 拍脑袋定超参，3次方；
            # 实际中需要试验
            return math.pow(np.dot(x1, x2) + 1, 3)
        elif type == 'gaussian':
            norm2square = math.pow(np.linalg(np.substract(x1, x2)), 2)
            # sigma的值也拍个脑袋吧=,=
            return math.exp(float(norm2square) / 2 / 10 * -1)
        else:
            return np.dot(x1, x2)

def main():
    training_set = np.array([[3.,3.],[4.,3.],[1.,1.]])
    training_labels = np.array([1,1,-1])
    svm = SVM(training_set, training_labels)
    svm.train()

    # should be 1
    print 'dot [5, 4] is ?'
    print svm.predict(np.array([5,4]))

def main2():
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values
    imgs = data[0::, 1::]
    labels = data[::, 0]
    new_labels = []
    for label in labels:
        new_labels.append(label * 2 - 1)
    # 用hog提取特征
    features = get_hog_features(imgs)

    # 拆分训练集，一部分用来测试
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, new_labels, test_size=0.33, random_state=23323)

    svm = SVM(train_features, train_labels)
    svm.train()

    predicts = []
    for x in test_features:
        predicts.append(svm.predict(x))

    score = accuracy_score(test_labels, predicts)
    print 'predict accuracy : ', score


def get_hog_features(trainset):
    hog = cv2.HOGDescriptor('../hog.xml')

    features = []
    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)
    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return features

if __name__ == '__main__':
    main2()

