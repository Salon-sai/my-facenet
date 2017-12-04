

import numpy as np
import cv2


def detect_face(image, minsize, pnet, rnet, onet, threshold, factor):
    """
    :param image: 输入照片
    :param minsize: 输出的最小人脸尺寸
    :param pnet: caffemodel
    :param rnet: caffemodel
    :param onet: caffemodel
    :param threshold: [th1, th2, th3] 三个模型的threshold
    :param factor: 将image resize的factor
    :return:
    """
    factor_count = 0
    total_boxes = np.empty((0, 9))
    points = np.empty(0)
    h = image.shape[0]
    w = image.shape[1]
    minl = np.amin([h, w])
    m = 12 / minsize

    minl = minl * m

    scales = []
    while minl >= 12:
        scales += [m*np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    for j in range(len(scales)):

        scale = scales[j]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(image, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0, 2, 1, 3))
        out = pnet(img_y)
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[0], (0, 2, 1, 3))
        boxes, _ = generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0])

        pick = nms(boxes.copy(), 0.5, 'Union')

        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick:]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick: ]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0: 4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)


def pad(total_boxes, w, h):
    # 计算padding大小（padding使得bounding boxes变成正方形）
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[: 2].copy().astype(np.int32)
    ey = total_boxes[: 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    x[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2: 4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxA

def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    return im_data


def generateBoundingBox(imap, reg, scale, t):

    stride = 2
    cellsize = 12
    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])

    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)

    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))

    if reg.size == 0:
        reg = np.empty((0, 3))
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    bounding_box = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return bounding_box, reg

def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)

        I = I[np.where(o < threshold)]
    pick = pick[0:counter]
    return pick

