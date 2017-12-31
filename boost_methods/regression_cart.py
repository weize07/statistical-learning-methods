#encoding=utf-8

# import pandas as pandas
import numpy as np
# import cv2
import math
import random
import time
import sys
from sklearn.metrics import r2_score

class RegressionCART(object):
    def __init__(self, inputs, labels, loss_func='square', max_level=10):
        self.root = TreeNode(inputs, labels, 1, loss_func, max_level)

    def train(self):
        self.root.train()

    def predict(self, x):
        return self.root.predict(x)


class TreeNode(object):
    def __init__(self, inputs, labels, level, loss_func='square', max_level=10):
        self.X = inputs
        self.N = len(inputs)
        # dimens of input
        self.D = len(self.X[0])
        self.Y = labels
        self.LF = loss_func
        self.C = None
        self.level = level
        self.L = max_level
        self.left = None
        self.right = None
        self.cutting_dimen = None
        self.cutting_point = None

    def _loss(self, Y, c):
        res = 0
        if self.LF == 'square':
            for i in range(len(Y)):
                y = Y[i]
                res += math.pow(y - c, 2) 
        return res

    def predict(self, x):
        if self.cutting_dimen == None:
            return self.C
        if x[self.cutting_dimen] <= self.cutting_point:
            if self.left != None:
                return self.left.predict(x)
            else:
                return self.C
        else:
            if self.right != None:
                return self.right.predict(x)
            else:
                return self.C

    def train(self):
        sum_y = 0
        for i in range(self.N):
            sum_y += self.Y[i]
        self.C = float(sum_y) / self.N

        if self.level == self.L:
            return
        dimen, point = self._get_cut_var_and_point(self.X)
        self.cutting_dimen = dimen
        self.cutting_point = point;

        left_X = []
        left_Y = []
        right_X = []
        right_Y = []
        for j in range(self.N):
            x = self.X[j]
            if x[dimen] <= point:
                left_X.append(self.X[j])
                left_Y.append(self.Y[j])
            else:
                right_X.append(self.X[j])
                right_Y.append(self.Y[j])
        if len(left_Y) == 0 or len(right_Y) == 0:
            return

        self.left = TreeNode(left_X, left_Y, self.level + 1, self.LF, self.L)
        self.left.train()
        self.right = TreeNode(right_X, right_Y, self.level + 1, self.LF, self.L)
        self.right.train()


    def _get_cut_var_and_point(self, inputs):
        dimen = -1
        point = -1
        value = -1
        min_loss = sys.maxint
        for d in range(self.D):
            for i in range(self.N):
                left_Y = []
                left_sum = 0
                right_Y = []
                right_sum = 0
                x = self.X[i]
                for j in range(self.N):
                    xx = self.X[j]
                    if xx[d] <= x[d]:
                        left_Y.append(self.Y[j])
                        left_sum += self.Y[j]
                    else:
                        right_Y.append(self.Y[j])
                        right_sum += self.Y[j]
                loss = 0
                if (len(right_Y) > 0):
                    loss += self._loss(right_Y, float(right_sum) / len(right_Y))
                if (len(left_Y) > 0):
                    loss += self._loss(left_Y, float(left_sum) / len(left_Y))
                if loss < min_loss:
                    min_loss = loss
                    point = x[d]
                    dimen = d
        return dimen, point

def main():
    fr = open('../data/ex0.txt')
    inputs = []
    labels = []
    for line in fr.readlines():
        splits = line.strip().split('\t')
        inputs.append([float(splits[0]), float(splits[1])])
        labels.append(float(splits[2]))
    r_cart = RegressionCART(inputs, labels, 'square', 2)
    r_cart.train()
    predicts = []
    for x in inputs:
        y = r_cart.predict(x)
        predicts.append(y)
        # print x, y
    score = r2_score(labels, predicts)
    print 'accuracy_score: ', score

if __name__ == '__main__':
    main()










