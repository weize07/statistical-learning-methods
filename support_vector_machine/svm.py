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
from generate_dataset import *

class SVM(object):
    """Summary of class here.

    Attributes:
        rate:       learning rate
        params of dual form svm: a and b;
        features:   training set, input vectors of features
        labels:     training set, labels of features;
    """

    def __init__(self, inputs, labels, soft_param=1000, error_thres=0.001, study_max=5000, nochange_max=50):
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
        print self.n
        self.dots = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                dot = self._kernel(self.inputs[i], self.inputs[j])
                self.dots[i][j] = dot
                self.dots[j][i] = dot
        pp.pprint(self.dots)
        self.e = np.zeros(self.n)
        for i in range(self.n):
            self.e[i] = self._error(i)
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
            # (i, j) = self._select2()
            i, j = self._select2()
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
            ei = self.e[i]
            ej = self.e[j]
            aj_new = self.a[j] + yj * (ei - ej) / \
                (self.dots[i][i] + self.dots[j][j] - 2 * self.dots[i][j])
            if aj_new > H:
                aj_new = H
            elif aj_new < L:
                aj_new = L
            ai_new = self.a[i] + yi * yj * (self.a[j] - aj_new)
            
            bi_new = - ei - yi * self.dots[i][i] * (ai_new - self.a[i]) \
                - yj * self.dots[j][i] * (aj_new - self.a[j]) + self.b
            bj_new = - ej - yi * self.dots[i][j] * (ai_new - self.a[i]) \
                - yj * self.dots[j][j] * (aj_new - self.a[j]) + self.b
            if ai_new > 0 and ai_new < self.C:
                self.b = bi_new
            elif aj_new > 0 and aj_new < self.C:
                self.b = bj_new
            else:
                self.b = float(bi_new + bj_new) / 2
            self.a[i] = ai_new
            self.a[j] = aj_new
            self.e[i] = self._error(i)
            self.e[j] = self._error(j)
        pp.pprint(self.a)
        pp.pprint(self.b)


    def predict(self, x):
        y = self._output(x)
        if y >= 0: 
            return 1
        else:
            return -1

    def _satisfy_KKT(self, i):
        ygx = self.labels[i] * self._cached_output(i)
        if abs(self.a[i]) < self.t:
            return ygx > 1 or ygx == 1
        elif abs(self.a[i]-self.C)<self.t:
            return ygx < 1 or ygx == 1
        else:
            return abs(ygx-1) < self.t


    def _select2(self):
        index_list = [i for i in xrange(self.n)]

        i1_list_1 = filter(lambda i: self.a[i] > 0 and self.a[i] < self.C, index_list)
        i1_list_2 = list(set(index_list) - set(i1_list_1))

        i1_list = i1_list_1
        i1_list.extend(i1_list_2)

        for i in i1_list:
            if self._satisfy_KKT(i):
                continue

            E1 = self.e[i]
            max_ = (0, 0)

            for j in index_list:
                if i == j:
                    continue

                E2 = self.e[j]
                if abs(E1 - E2) > max_[0]:
                    max_ = (abs(E1 - E2), j)

            return i, max_[1]
        return -1, -1


    def _select(self):
        (res1, res2) = (-1, -1)

        for i in range(self.n):
            if self.a[i] < self.C and self.a[i] > 0:
                if (math.fabs(self.labels[i] * self._cached_output(i) - 1) > self.t):
                    res1 = i
                    break
        if res1 == -1:
            for i in range(self.n):
                if self.a[i] == 0:
                    if (self.labels[i] * self._cached_output(i) <= 1 - self.t):
                        res1 = i
                        break
                elif self.a[i] == self.C:
                    if (self.labels[i] * self._cached_output(i) >= 1 + self.t):
                        res1 = i
                        break
        if res1 == -1:
            return (res1, res2)

        error1 = self.e[res1]
        maxE = -1
        res2 = -1
        for i in range(self.n):
            if i == res1:
                continue
            error2 = self.e[i]
            ediff = math.fabs(error2 - error1)
            if ediff >  maxE:
                res2 = i
                maxE = ediff
        return (res1, res2)

    def _cached_output(self, i):
        dotsum = 0
        for j in range(self.n):
            dotsum += self.dots[i][j] * self.labels[j] * self.a[j]
        return dotsum + self.b

    def _output(self, x):
        dotsum = 0
        for i in range(self.n):
            dotsum += self._kernel(x, self.inputs[i]) * self.labels[i] * self.a[i]
        return dotsum + self.b

    def _error(self, i):
        return self._cached_output(i) - self.labels[i];


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
    print 'dot [3.01, 1] is ?'
    print svm.predict(np.array([3.01, 1]))
    print 'dot [2, 1.99] is ?'
    print svm.predict(np.array([2, 1.99]))


def main2():
    # raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    # data = raw_data.values
    # imgs = data[0::, 1::]
    # labels = data[::, 0]
    # new_labels = []
    # for label in labels:
    #     new_labels.append(label * 2 - 1)

    # # 用hog提取特征
    # features = get_hog_features(imgs)
    print "test"

    # 拆分训练集，一部分用来测试
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, train_labels, test_features, test_labels = generate_dataset(2000, visualization=False)
    # train_features, test_features, train_labels, test_labels = \
    #     train_test_split(features, new_labels, test_size=0.33, random_state=23323)
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    print "test0"
    svm = SVM(train_features, train_labels)
    print "test1"
    svm.train()

    predicts = []
    for x in test_features:
        predicts.append(svm.predict(x))
    print "test2"

    score = accuracy_score(test_labels, predicts)
    print 'predict accuracy : ', score

def main3():
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values
    imgs = data[0::, 1::]
    labels = data[::, 0]
    new_labels = []
    for label in labels:
        new_labels.append(label * 2 - 1)
        if len(new_labels) == 4000:
            break

    # 用hog提取特征
    features = get_hog_features(imgs)
    print "test"

    # 拆分训练集，一部分用来测试
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, new_labels, test_size=0.33, random_state=23323)

    print "test0"
    svm = SVM(train_features, train_labels)
    print "test1"
    svm.train()

    predicts = []
    for x in test_features:
        predicts.append(svm.predict(x))
    print "test2"

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
        if len(features) == 4000:
            break
    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    print len(features)
    return features

if __name__ == '__main__':
    main()
    # main2()
    main3()

