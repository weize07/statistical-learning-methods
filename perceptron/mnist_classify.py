#encoding=utf-8

import pandas as pd
import numpy as np
import cv2

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from binary_perceptron import BinaryPerceptron

def main():
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

    perceptron = BinaryPerceptron(train_features, train_labels, 0.0001)
    perceptron.train()

    predicts = []
    for x in test_features:
    	predicts.append(perceptron.predict(x))

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
    main()
    



