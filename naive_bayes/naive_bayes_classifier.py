#encoding=utf-8

import random
import time

class NaiveBayesClassifier(object):
    def __init__(self, inputs, labels, features, classes, smoothing=1):
        self.training_inputs = inputs
        self.training_labels = labels
        self.features = features
        self.classes = classes
        self.smoothing = smoothing


    # With Laplace smoothing
    def train(self):
        self.class_probs = {}
        self.feature_condition_probs_class = {}

        class_counts = {'total': self.smoothing * len(self.classes)}
        for c in self.classes:
            class_counts[c] = self.smoothing
        for x in self.training_labels:
            class_counts[x] += 1
            class_counts['total'] += 1
        for c in class_counts.keys():
            self.class_probs[c] = float(class_counts[c]) / class_counts['total']

        class_feature_counts = {}
        for c in self.classes:
            class_feature_counts[c] = {}
            for dimen in range(len(self.features)):
                cf = self.features[dimen]
                class_feature_counts[c][dimen] = {'total': self.smoothing * len(cf)}
                for cfv in cf:
                    class_feature_counts[c][dimen][cfv] = self.smoothing
        for index in range(len(self.training_labels)):
            l = self.training_labels[index]
            x = self.training_inputs[index]
            for i in range(len(x)):
                xi = x[i]
                class_feature_counts[l][i][xi] += 1
                class_feature_counts[l][i]['total'] += 1


        # print class_feature_counts        

        for c in class_feature_counts:
            values = class_feature_counts[c]
            self.feature_condition_probs_class[c] = {}
            for dimen in values:
                counts = values[dimen]
                self.feature_condition_probs_class[c][dimen] = {}
                for index in counts:
                    if index == 'total':
                        continue
                    value = counts[index]
                    self.feature_condition_probs_class[c][dimen][index] = float(value) / counts['total']
        print self.feature_condition_probs_class        



    def predict(self, x):
        max_prob = -1
        max_class = None
        for c in self.classes:
            prob = self.class_probs[c]
            print 'class ', c, ' priori prob is ', prob
            for index in range(len(x)):
                xi = x[index]
                prob *= self.feature_condition_probs_class[c][index][xi]
            print x, ' joint prob on class ', c, ' is ', prob
            if prob > max_prob:
                max_prob = prob
                max_class = c

        return max_class


def main():
    training_set = [[1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'], \
        [2,'S'],[2,'M'],[2,'M'],[2,'L'],[2,'L'], \
        [3,'L'],[3,'M'],[3,'M'],[3,'L'],[3,'L']]
    training_labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    features = [[1, 2, 3],['S','M','L']]
    classes = [1, -1]
    nbc = NaiveBayesClassifier(training_set, training_labels, features, classes)
    nbc.train()

    # should be -1
    print '[2, S] is ?'
    print nbc.predict([2, 'S'])

if __name__ == '__main__':
    main()





