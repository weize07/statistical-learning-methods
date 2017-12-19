#encoding=utf-8

import pprint as pp
import math

class ID3DecisionTree(object):
    def __init__(self, inputs, labels, features, classes, info_gain_thres):
        self.training_inputs = inputs
        self.training_labels = labels
        self.features = features
        self.classes = classes
        self.info_gain_thres = info_gain_thres

    def train(self):
        self.tree = TreeNode(self.training_inputs, self.training_labels, self.features, self.classes, None, None)
        self.tree.split(self.info_gain_thres)

    def predict(self, input):
        return


class TreeNode(object):
    def __init__(self, inputs, labels, features, classes, feature_id, value):
        self.inputs = inputs
        self.labels = labels
        self.features = features
        self.classes = classes
        self.branches = []
        self.feature_id = feature_id
        self.value=value
        # 统计labels中最多的label，作为当前节点的label

        self.entropy=self._entropy(labels)


    def split(self, info_gain_thres):
        print 'entropy', self.entropy
        for feature_id in range(len(self.features)):
            ig = self._info_gain(feature_id)
            print 'info_gain', ig
        # 第一步，计算所有feature的信息增益，选出最大的，假设是featureA
        # 如果信息增益小于info_gain_thres，或者没有可用的feature了，直接返回

        # 第二步，假设featureA有k个值，那么dataList拆分成k个，labels拆分成k个，
        # 从features中剔除featureA，并建立k个子节点，加入self.branches中
        
        # 第三步，递归调用branches的split.


    def _info_gain(self, feature_id):
        return self.entropy - self._conditional_entropy(feature_id)

    def _entropy(self, labels):
        count = len(labels)
        counts = {}
        for label in labels:
            if counts.has_key(label):
                counts[label] += 1;
            else:
                counts[label] = 1;

        res = 0
        for label in counts:
            prob = float(counts[label]) / count
            if prob != 0:
                res -= prob * math.log(prob, 2)
        return res

    def _conditional_entropy(self, feature_id):
        values = self.features[feature_id]
        feature_value_counts = {}
        for value in values:
            feature_value_counts[value] = {'total':0}
            for c in self.classes:
                feature_value_counts[value][c] = 0

        res = 0
        for i in range(len(self.inputs)):
            x = self.inputs[i]
            c = self.labels[i]
            feature_value_counts[x[feature_id]][c] += 1
            feature_value_counts[x[feature_id]]['total'] += 1

        res = 0
        for value in feature_value_counts:
            value_counts = feature_value_counts[value]
            tmp = 0
            for c in value_counts:
                if c == 'total':
                    continue;
                prob = float(value_counts[c]) / value_counts['total']
                if prob != 0:
                    tmp -= prob * math.log(prob, 2)
            percentage = float(value_counts['total']) / len(self.inputs)
            res += percentage * tmp
        print 'conditional_entropy of ', feature_id, '=', res
        return res


def main():
    """
    # features: 
    age         (0: young, 1: mid, 2: old)
    work        (0: working, 1: Dally)
    has_house   (0: yes, 1: no)
    loan        (0: normal, 1: good, 2: excellent)

    # classes[whether loan]: (0: yes, 1: no)
    """
    training_inputs = [[0, 1, 1, 0],[0, 1, 1, 1],[0, 0, 1, 1],[0, 0, 0, 0],[0, 1, 1, 0], \
        [1, 1, 1, 0],[1, 1, 1, 1],[1, 0, 0, 1],[1, 1, 0, 2],[1, 1, 0, 2], \
        [2, 1, 0, 2],[2, 1, 0, 1],[2, 0, 1, 1],[2, 0, 1, 2],[2, 1, 1, 0]]
    training_labels = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    features = [[0, 1, 2],[0, 1],[0, 1],[0, 1, 2]]
    classes = [0, 1]
    id3dt = ID3DecisionTree(training_inputs, training_labels, features, classes, 0)
    print id3dt.train()

    

if __name__ == '__main__':
    main()


