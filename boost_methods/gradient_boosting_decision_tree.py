#encoding=utf-8

import numpy as np
import pandas as pd
import numpy as np
import random
import time
from regression_cart import RegressionCART
from sklearn.metrics import r2_score

class GBDT(object):
    def __init__(self, inputs, labels, loss_func='square', max_tree_num=10):
        self.X = inputs
        self.N = len(inputs)
        # dimens of input
        self.Y = labels
        self.LF = loss_func
        self.M = max_tree_num
        self.dts = []


    def _loss(self, Y, c):
        res = 0
        if self.LF == 'square':
            for i in range(len(X)):
                y = Y[i]
                res += math.square(y - c) 
        return res


    def train(self):
        # epsilon = 0.0001
        count = 0
        outputs = self.Y
        while count < self.M:
            count += 1
            dt = RegressionCART(self.X, outputs, self.LF, 10)
            dt.train()
            self.dts.append(dt)
            FX = []
            for i in range(self.N):
                x = self.X[i]
                FX.append(dt.predict(x))
            # 计算残差
            outputs = self._gradient(outputs, FX)
            # stop = True
            # for i in outputs:
            #     if i > epsilon:
            #         stop = False
            # if stop:
            #     break


    def predict(self, x):
        res = 0
        i = 0
        for dt in self.dts:
            # print i 
            tmp = dt.predict(x)
            # print tmp
            res += tmp
        return res


    # 所用损失函数对应的残差计算，用负梯度表示
    def _gradient(self, Y, FX):
        # rmi = - DELTA( L(yi, f(xi)) ) / DELTA( f(xi) )
        res = []
        if self.LF == 'square':
            for i in range(self.N):
                res.append(Y[i] - FX[i])
        return res

def main():
    fr = open('../data/ex0.txt')
    inputs = []
    labels = []
    for line in fr.readlines():
        splits = line.strip().split('\t')
        inputs.append([float(splits[0]), float(splits[1])])
        labels.append(float(splits[2]))
    gbdt = GBDT(inputs, labels, 'square', 2)
    gbdt.train()
    predicts = []
    for x in inputs:
        y = gbdt.predict(x)
        predicts.append(y)
        # print x, y
    score = r2_score(labels, predicts)
    print 'accuracy_score: ', score
    

if __name__ == '__main__':
    main()

