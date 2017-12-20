#encoding=utf-8

import math
import numpy as np

class BinaryLogisticRegression(object):
    def __init__(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        # extend w to [w, b], x to [x, 1]
        self.weight = np.zeros(len(self.train_inputs[0]) + 1) 
        # self.weight = np.array([-1.87185636, 11.20587194, 10.13296618, -10.41420202  -4.40042467])

    def train(self, max_study=10000):
        while (max_study > 0):
            max_study -= 1
            (loss, gradient) = self._loss()
            self.weight = np.add(self.weight, gradient)
        print self.weight

    def _loss(self):
        loss = 0
        gradient = np.zeros(len(self.train_inputs[0]) + 1) 
        for i in range(len(self.train_inputs)):
            x = np.append(self.train_inputs[i], [1])
            y = self.train_labels[i]
            wx = np.dot(self.weight, x)
            e = math.exp(wx)
            loss -= (y * wx - math.log(1 + e))
            # print (1 - e)
            tmp = y * x - np.divide(e * x, 1 + e)
            gradient = np.add(gradient, tmp)
        return (loss, gradient)


    def predict(self, x):
        x = np.append(x, [1])
        e = math.exp(np.dot(self.weight, x))
        neg_probability = e / (1 + e)
        pos_probability = 1 - neg_probability
        return (pos_probability, neg_probability)

def main():
    """
    # features: 
    0 age         (0: young, 1: mid, 2: old)
    1 work        (0: working, 1: Dally)
    2 has_house   (0: yes, 1: no)
    3 loan        (0: normal, 1: good, 2: excellent)

    # classes[whether loan]: (0: yes, 1: no)
    """
    training_inputs = [[0, 1, 1, 0],[0, 1, 1, 1],[0, 0, 1, 1],[0, 0, 0, 0],[0, 1, 1, 0], \
        [1, 1, 1, 0],[1, 1, 1, 1],[1, 0, 0, 1],[1, 1, 0, 2],[1, 1, 0, 2], \
        [2, 1, 0, 2],[2, 1, 0, 1],[2, 0, 1, 1],[2, 0, 1, 2],[2, 1, 1, 0]]
    training_labels = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    sigmoid = BinaryLogisticRegression(training_inputs, training_labels)
    sigmoid.train(1000)
    print 'train finish, final loss : ', sigmoid._loss()
    print '---------------------------'
    (pos, neg) = sigmoid.predict([0, 1, 1, 0])
    print 'loan to [0, 1, 1, 0]? positive :', pos, 'negative :', neg
    (pos, neg) = sigmoid.predict([0, 1, 0, 2])
    print 'loan to [0, 1, 0, 2]? positive :', pos, 'negative :', neg


if __name__ == '__main__':
    main()

