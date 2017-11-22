# -*- coding: UTF-8 -*-

import numpy as np
from scipy import interpolate
from sklearn.model_selection import KFold

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # 正负判断的threshold
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame))

    thresholds = np.arange(0, 4, 0.001)


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[0] == len(actual_issame)
    assert embeddings1.shape[1] == embeddings2.shape[1]

    nrof_thresholds = len(thresholds)
    nrof_pairs = len(actual_issame)
    kf = KFold(n_splits=nrof_folds, shuffle=True)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), axis=1)
    indices = np.arange(nrof_pairs)
    for kfold_index, (train_index, test_index) in enumerate(kf.split(indices)):
        train_acc = np.zeros((nrof_thresholds))
        for threshold_index, threshold in enumerate(thresholds):
            _, _, train_acc[threshold_index] = calculate_accuracy(threshold,
                                                                  dist[train_index],
                                                                  actual_issame[train_index])
        best_threshold_index = np.argmax(train_acc)
        best_thresholds[kfold_index] = thresholds[best_threshold_index]
        for threshold_index, threshold in enumerate(thresholds):
            if threshold_index == best_threshold_index:
                tprs[kfold_index, threshold_index], fprs[kfold_index, threshold_index], accuracy[kfold_index] = \
                    calculate_accuracy(threshold, dist[test_index], actual_issame[test_index])
            else:
                tprs[kfold_index, threshold_index], fprs[kfold_index, threshold_index], _ = \
                    calculate_accuracy(threshold, dist[test_index], actual_issame[test_index])


    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]

    nrof_thresholds = len(thresholds)
    nrof_pairs = len(actual_issame)
    kf = KFold(n_splits=nrof_folds, shuffle=True)

    vals = np.zeros(nrof_folds)
    fars = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_index, (train_index, test_index) in enumerate(kf.split(indices)):

        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_index], actual_issame[train_index])

        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0

        vals[fold_index], fars[fold_index] = calculate_val_far(threshold, dist[test_index], actual_issame[test_index])

    val_mean = np.mean(vals)
    far_mean = np.mean(fars)
    val_std = np.std(vals)
    return val_mean, val_std, far_mean


def calculate_accuracy(threshold, dist, actual_issame):
    """
    计算准确率：真准确率，假准确率，误报率，漏报率
    True Positive Rate（真正率）/灵敏度
    False Negative Rate（真负率）/特指度
    :param threshold: 判断阀门
    :param dist:
    :param actual_issame:
    :return:
    """
    predict_issame = np.less(dist, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn) == 0 else float(tp) / tp + fn
    fpr = 0 if (tn + fp) == 0 else float(tn) / tn + fp
    accuarcy = float(tp + tn) / len(dist)

    return tpr, fpr, accuarcy

def calculate_val_far(threshold, dist, actual_issame):
    """

    :param threshold:
    :param dist:
    :param actual_issame:
    :return:
    """
    predict_issame = np.less(dist, threshold)
    true_accpet = np.logical_and(predict_issame, actual_issame)
    false_accpet= np.logical_and(predict_issame, np.logical_not(actual_issame))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accpet) / float(n_same)
    far = float(false_accpet) / float(n_diff)
    return val, far
