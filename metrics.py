# -*- coding: UTF-8 -*-

import numpy as np
from scipy import interpolate
from sklearn.model_selection import KFold

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # 正负判断的threshold
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, threshold_accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                                           np.asarray(actual_issame), nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, far, threshold_var_far = calculate_val(thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
                                                1e-3, nrof_folds)

    return tpr, fpr, accuracy, val, threshold_var_far, far, threshold_accuracy

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    """
    :param thresholds: 多个阈值取值
    :param embeddings1: 校验样本的第一个embedding
    :param embeddings2: 校验样本的第二个embedding
    :param actual_issame: 样本是否为同一个类别
    :param nrof_folds: kfold的分批量
    :return:
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[0] == len(actual_issame)
    assert embeddings1.shape[1] == embeddings2.shape[1]

    # nrof_thresholds = len(thresholds)
    # nrof_pairs = len(actual_issame)
    # kf = KFold(n_splits=nrof_folds, shuffle=True)
    #
    # tprs = np.zeros((nrof_folds, nrof_thresholds))
    # fprs = np.zeros((nrof_folds, nrof_thresholds))
    # accuracy = np.zeros((nrof_folds))
    # best_thresholds = np.zeros((nrof_folds))
    #
    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), axis=1)
    # indices = np.arange(nrof_pairs)
    # for kfold_index, (train_index, test_index) in enumerate(kf.split(indices)):
    #     train_acc = np.zeros((nrof_thresholds))
    #     for threshold_index, threshold in enumerate(thresholds):
    #         _, _, train_acc[threshold_index] = calculate_accuracy(threshold,
    #                                                               dist[train_index],
    #                                                               actual_issame[train_index])
    #     best_threshold_index = np.argmax(train_acc)
    #     best_thresholds[kfold_index] = thresholds[best_threshold_index]
    #     for threshold_index, threshold in enumerate(thresholds):
    #         if threshold_index == best_threshold_index:
    #             tprs[kfold_index, threshold_index], fprs[kfold_index, threshold_index], accuracy[kfold_index] = \
    #                 calculate_accuracy(threshold, dist[test_index], actual_issame[test_index])
    #         else:
    #             tprs[kfold_index, threshold_index], fprs[kfold_index, threshold_index], _ = \
    #                 calculate_accuracy(threshold, dist[test_index], actual_issame[test_index])
    #
    #
    # tpr = np.mean(tprs, 0)
    # fpr = np.mean(fprs, 0)

    nrof_threholds = len(thresholds)
    tpr = np.zeros(nrof_threholds)
    fpr = np.zeros(nrof_threholds)
    accuracy = np.zeros(nrof_threholds)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    for index, threshold in enumerate(thresholds):
        tpr[index], fpr[index], accuracy[index] = calculate_accuracy(threshold, dist, actual_issame)
    return tpr, fpr, accuracy, thresholds[np.argmax(accuracy)]

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    """

    :param thresholds:
    :param embeddings1:
    :param embeddings2:
    :param actual_issame:
    :param far_target:
    :param nrof_folds:
    :return:
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]

    # nrof_thresholds = len(thresholds)
    # nrof_pairs = len(actual_issame)
    # kf = KFold(n_splits=nrof_folds, shuffle=True)
    #
    # vals = np.zeros(nrof_folds)
    # fars = np.zeros(nrof_folds)
    #
    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    # indices = np.arange(nrof_pairs)
    #
    # for fold_index, (train_index, test_index) in enumerate(kf.split(indices)):
    #
    #     far_train = np.zeros(nrof_thresholds)
    #     for threshold_idx, threshold in enumerate(thresholds):
    #         _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_index], actual_issame[train_index])
    #
    #     # 判断本次k-fold中，所有的far是否低于目标值。若是存在高于目标值，则进行插值处理并根据生成的插值函数重新计算阈值
    #     if np.max(far_train) >= far_target:
    #         f = interpolate.interp1d(far_train, thresholds, kind='slinear')
    #         threshold = f(far_target)
    #     else:
    #         threshold = 0
    #
    #     # 本次fold中，使用最后阈值计算出val,far值
    #     vals[fold_index], fars[fold_index] = calculate_val_far(threshold, dist[test_index], actual_issame[test_index])
    #
    # val_mean = np.mean(vals)
    # far_mean = np.mean(fars)
    # val_std = np.std(vals)

    nrof_thresholds = len(thresholds)
    vals = np.zeros(nrof_thresholds)
    fars = np.zeros(nrof_thresholds)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    for index, threshold in enumerate(thresholds):
        vals[index], fars[index] = calculate_val_far(threshold, dist, actual_issame)
    # if np.max(fars) >= far_target:
    #     f = interpolate.interp1d(fars, thresholds, kind='slinear')
    #     threshold = f(far_target)
    # else:
    #     threshold = 0

    best_threshold = thresholds[np.argmin(fars)]

    # val_mean = np.mean(vals)
    # far_mean = np.mean(fars)
    # val_std = np.std(vals)
    val, far = calculate_val_far(best_threshold, dist, actual_issame)
    return val, far, best_threshold


def calculate_accuracy(threshold, dist, actual_issame):
    """
    计算准确率：真准确率，假准确率，误报率，漏报率
    True Positive Rate（真正率）/灵敏度
    False Negative Rate（真负率）/特指度
    :param threshold: 判断阈值
    :param dist:
    :param actual_issame:
    :return:
    """
    predict_issame = np.less(dist, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # print("threshold: %.4f\t tp: %d\t fp: %d\t tn: %d\t fn: %d" % (threshold, int(tp), int(fp), int(tn), int(fn)))
    tpr = 0 if (tp + fn) == 0 else float(tp) / float(tp + fn)
    fpr = 0 if (tn + fp) == 0 else float(tn) / float(tn + fp)
    accuarcy = float(tp + tn) / len(dist)
    return tpr, fpr, accuarcy

def calculate_val_far(threshold, dist, actual_issame):
    """
    计算机正确的接受比例和错误接受比例
    :param threshold: 阈值
    :param dist: 样本中图片间的距离
    :param actual_issame: 样本时候相同
    :return:
    val : 正确接受比例（正确判断是同一个人的样本数量/相同的类别的总数）
    far : 错误接受比例（错误判断是同一个人的样本数量/不同的类别的总数）
    """
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept= np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print("threshold: %.4f\t true_accept: %d\t false_accept: %d\t n_same: %d\t n_diff: %d" % \
    #       (threshold, int(true_accept), int(false_accept), int(n_same), int(n_diff)))
    val = 0 if n_same == 0 else float(true_accept) / float(n_same)
    far = 0 if n_diff == 0 else float(false_accept) / float(n_diff)
    return val, far
