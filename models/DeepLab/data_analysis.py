import PIL
import numpy as np
import os
import argparse
import shutil
import sys
import cv2
from os import listdir
from os.path import join
import tensorflow as tf
from sklearn.metrics import confusion_matrix


MAP_P2L = '/media/workspace/rueisung/TFrecord/512/valid_name_number_512.npy'
LABEL = '/media/workspace/bgong/data/road-extraction/train'
PREDICTION = '/home/chenny1229/parameters/512/baseline_augment'


def main():

    map_p2l = np.load(MAP_P2L)
    pred_dir = PREDICTION
    label_dir = LABEL

    num = len(map_p2l)
    metric_map = {}
    for i in range(num):
        print('reading', i, '...')
        prediction = cv2.imread(join(pred_dir, '%04d_prob.png'%i))
        
        # label = cv2.imread(join(label_dir, map_p2l[i]+'_mask.png'))
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # label = cv2.resize(label, (512,512),interpolation=cv2.INTER_AREA)
        
        b,label,r = cv2.split(cv2.imread(join(pred_dir, '%04d_mask.png'%i)))
        label = label[:,512:]
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
        prediction = prediction.flatten()
        label = label.flatten()
        label = np.where(np.less_equal(label, 128), np.zeros_like(label), np.ones_like(label))
        for threshold in range(30, 101, 10):
            metric_map['miou_' + str(threshold)] = 0
            metric_map['precision_' + str(threshold)] = 0
            metric_map['recall_' + str(threshold)] = 0

        for threshold in range(30, 101, 10):
            temp = np.where(np.less_equal(prediction, threshold), 
                   np.zeros_like(prediction), np.ones_like(prediction))
            cm = confusion_matrix(temp, label)
            metric_map['miou_' + str(threshold)] += cm[1, 1] / (cm[0, 1] + cm[1, 0] + cm[1, 1])
            metric_map['precision_' + str(threshold)] += cm[1, 1] / (cm[1, 0] + cm[1, 1])
            metric_map['recall_' + str(threshold)] += cm[1, 1] / (cm[0, 1] + cm[1, 1])
    
    print('sum up...')
    for threshold in range(50, 101, 10):
        print(threshold,": miou", metric_map['miou_' + str(threshold)],"precision",
                metric_map['precision_' + str(threshold)],"recall",metric_map['recall_' + str(threshold)])

    np.save('miou_recall_precision', metric_map)

if __name__ == '__main__':
    main()
