#encoding=utf-8

# import pandas as pandas
import numpy as np
# import cv2
import random
import time

class BinaryPerceptron(object):
    """Summary of class here.

    Attributes:
        rate:       learning rate
        w:          weight vector
        b:          bias scalar
        features:   training set, input vectors of features
        labels:     training set, labels of features;
    """

    def __init__(self, features, labels, rate, study_max=10000, nochange_max=50):
        """Inits BinaryPerceptron."""
        self.features = features
        self.labels = labels
        self.w = np.zeros(features[0].size)
        self.b = 0
        self.rate = rate
        self.study_max = study_max
        self.nochange_max = nochange_max

    def train(self):
        """learn w and b from training set."""
        set_size = len(self.features)
        nochange_count = 0
        study_count = 0
        while True:
            if (study_count >= self.study_max 
                or nochange_count >= self.nochange_max):
                break;
            index = random.randint(0, set_size - 1)
            x = self.features[index]
            y = self.labels[index]
            res = y * (np.dot(x, self.w) + self.b)
            if res <= 0:
                self.w = self.w + y * x * self.rate
                self.b = self.b + y * self.rate
                nochange_count = 0
                study_count += 1
            else:
                nochange_count += 1
        print '--- training finished ---'
        print 'weight: '
        print self.w
        print 'bias: '
        print self.b
        print '---------------'

    def predict(self, input):
        res = np.dot(input, self.w) + self.b
        if res <= 0:
            return -1
        else:
            return 1


def main():
    training_set = np.array([[3.,3.],[4.,3.],[1.,1.]])
    training_labels = np.array([1,1,-1])
    perceptron = BinaryPerceptron(training_set, training_labels, 1)
    perceptron.train()

    # should be 1
    print 'dot [5, 4] is ?'
    print perceptron.predict(np.array([5,4]))

if __name__ == '__main__':
    main()


