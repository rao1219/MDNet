__author__ = 'stephen'

####################################
# Author: Zilin Zhang
# Organization: Zhejiang University
# City: Hangzhou
# Description: The file contains
# multiple sample methods
####################################
# Generate samples based on bbox
#     :arg
#     im: cv2's image
#     bbox: ground-truth box(x, y, w, h)
#     params: five-tuple(width, height, scale, pos_threshold, neg_threshold) of gaussian parameters
#     num: number of samples
#
#     :return
#     bboxes: list of boxes
#             {
#                 'img' :img,
#                 'box'(x, y, w, h),
#                 'label': label,
#                 'overlap': overlap
#             }
####################################

import cv2
import numpy as np
import numpy.random as npr
from numpy.random import randn
from lib.utils.bbox import bbox_overlaps


def uniform_aspect_sample(im, bbox, params, num, stype):
    assert len(bbox) == 4, "Invalid ground-truth(x, y, w, h) form."
    assert bbox[2] > 0 and bbox[3] > 0, "Width or height < 0."
    assert len(params) == 5, "Invalid {:d}-tuple params(should be five-tuple).".format(len(params))
    assert num > 0, "Number of samples should be larger than 0."

    im_shape = im.shape
    im_w = im_shape[1]
    im_h = im_shape[0]

    # Calculate average of width and height
    centerx = bbox[0] + bbox[2] / 2.
    centery = bbox[1] + bbox[3] / 2.

    xrand = params[0] * bbox[2] * (npr.rand(num, 1) * 2 - 1)
    yrand = params[1] * bbox[3] * (npr.rand(num, 1) * 2 - 1)
    wrand = bbox[2] * (1.05 ** (npr.rand(num, 1) * 4 - 2))
    hrand = bbox[3] * (1.05 ** (npr.rand(num, 1) * 4 - 2))
    ws = wrand * (1.05 ** npr.rand(num, 1))
    hs = hrand * (1.05 ** npr.rand(num, 1))

    bboxes = []
    for i in range(num):
        cx = centerx + xrand[i, 0]
        cy = centery + yrand[i, 0]
        hw = ws[i, 0] / 2.
        hh = hs[i, 0] / 2.
        box = (
            max(0, int(cx - hw)),
            max(0, int(cy - hh)),
            min(im_w, int(cx + hw)),
            min(im_h, int(cy + hh))
        )
        if box[0] == box[2] or box[1] == box[3]:
            continue
        sample = (box[0], box[1], box[2] - box[0], box[3] - box[1])
        overlap = bbox_overlaps([bbox], [sample])[0]
        if overlap > params[3]:
            bboxes.append({
                'img': im,
                'box': sample,
                'label': 1,
                'overlap': overlap
            })
        elif overlap < params[4]:
            bboxes.append({
                'img': im,
                'box': sample,
                'label': 0,
                'overlap': overlap
            })
    return bboxes


def uniform_sample(im, bbox, params, num, stype):
    assert len(bbox) == 4, "Invalid ground-truth(x, y, w, h) form."
    assert bbox[2] > 0 and bbox[3] > 0, "Width or height < 0."
    assert len(params) == 5, "Invalid {:d}-tuple params(should be five-tuple).".format(len(params))
    assert num > 0, "Number of samples should be larger than 0."

    im_shape = im.shape
    im_w = im_shape[1]
    im_h = im_shape[0]

    # Calculate average of width and height
    centerx = bbox[0] + bbox[2] / 2
    centery = bbox[1] + bbox[3] / 2

    mean = round((bbox[2] + bbox[3]) / 2.)
    xrand = params[0] * mean * npr.rand(num, 1)*2 - 1
    yrand = params[1] * mean * npr.rand(num, 1)*2 - 1
    srand = 1.05 ** ((npr.rand(num, 1)*2 - 1) * params[3])
    bboxes = []
    for i in range(num):
        cx = centerx + xrand[i, 0]
        cy = centery + yrand[i, 0]
        hw = bbox[2] * srand[i, 0] / 2.
        hh = bbox[3] * srand[i, 0] / 2.
        box = (
            max(0, int(cx - hw)),
            max(0, int(cy - hh)),
            min(im_w, int(cx + hw)),
            min(im_h, int(cy + hh))
        )
        if box[0] == box[2] or box[1] == box[3]:
            continue
        sample = (box[0], box[1], box[2] - box[0], box[3] - box[1])
        overlap = bbox_overlaps([bbox], [sample])[0]
        if overlap > params[3]:
            bboxes.append({
                'img': im,
                'box': sample,
                'label': 1,
                'overlap': overlap
            })
        elif overlap < params[4]:
            bboxes.append({
                'img': im,
                'box': sample,
                'label': 0,
                'overlap': overlap
            })
    return bboxes


def gaussian_sample(im, bbox, params, num, stype):
    assert len(bbox) == 4, "Invalid ground-truth(x, y, w, h) form."
    assert bbox[2] > 0 and bbox[3] > 0, "Width or height < 0."
    assert len(params) == 5, "Invalid {:d}-tuple params(should be five-tuple).".format(len(params))
    assert num > 0, "Number of samples should be larger than 0."

    im_shape = im.shape
    im_w = im_shape[1]
    im_h = im_shape[0]

    # Calculate average of width and height
    centerx = bbox[0] + bbox[2] / 2
    centery = bbox[1] + bbox[3] / 2

    ones = np.ones((num, 1))
    neg_ones = -1 * ones

    mean = round((bbox[2] + bbox[3]) / 2.)
    min_ = np.min(np.hstack((ones, 0.5 * randn(num, 1))), axis=1)
    min_ = min_.reshape((min_.size, 1))
    max_ = np.max(np.hstack((neg_ones, min_)), axis=1)
    offsetx = params[0] * mean * max_
    min_ = np.min(np.hstack((ones, 0.5 * randn(num, 1))), axis=1)
    min_ = min_.reshape((min_.size, 1))
    max_ = np.max(np.hstack((neg_ones, min_)), axis=1)
    offsety = params[1] * mean * max_

    min_ = np.min(np.hstack((ones, 0.5 * randn(num, 1))), axis=1)
    min_ = min_.reshape((min_.size, 1))
    max_ = params[2] * np.max(np.hstack((neg_ones, min_)), axis=1)
    scale = 1.05 ** max_

    w = (bbox[2] * scale)[:, np.newaxis]
    h = (bbox[3] * scale)[:, np.newaxis]
    tens = np.array([10] * num)[:, np.newaxis]
    w_minus_10 = np.array(w - 10)
    h_minus_10 = np.array(h - 10)
    if stype == 'TRAIN':
        wmin_ = np.min(np.hstack((w_minus_10, w)), axis=1)[:, np.newaxis]
        hmin_ = np.min(np.hstack((h_minus_10, h)), axis=1)[:, np.newaxis]
        ws = np.max(np.hstack((tens, wmin_)), axis=1)
        hs = np.max(np.hstack((tens, hmin_)), axis=1)
    elif stype == 'TEST':
        ws = np.max(np.hstack((tens, w)), axis=1)
        hs = np.max(np.hstack((tens, h)), axis=1)
    bboxes = []
    for i in range(num):
        hw = ws[i] / 2
        hh = hs[i] / 2
        box = (
            max(0, int(centerx + offsetx[i] - hw)),
            max(0, int(centery + offsety[i] - hh)),
            min(im_w, int(centerx + offsetx[i] + hw)),
            min(im_h, int(centery + offsety[i] + hh))
        )
        if box[0] == box[2] or box[1] == box[3]:
            continue
        sample = (box[0], box[1], box[2] - box[0], box[3] - box[1])
        overlap = bbox_overlaps([bbox], [sample])[0]
        if overlap > params[3]:
                bboxes.append({
                    'img': im,
                    'box': sample,
                    'label': 1,
                    'overlap': overlap
                })
        elif overlap < params[4]:
                bboxes.append({
                    'img': im,
                    'box': sample,
                    'label': 0,
                    'overlap': overlap
                })
    return bboxes


if __name__ == '__main__':
    """Experiment code for testing.
    """
    # from lib.utils.image import im_bbox_show
    from dataset_factory import VDBC

    vdbc = VDBC(dbtype='OTB',
                gtpath='D:\\dataset\\OTB',
                dbpath='D:\\dataset\\OTB',
                flush=True)

    im_list, gts, fd_map = vdbc.get_db()
    im_path = im_list[fd_map[2]][11]
    im = cv2.imread(im_path)
    gt = gts[fd_map[2]][11]

    bboxes = uniform_aspect_sample(im, gt, (0.3, 0.3, 10, 0.7, 0.5), 1000, stype='TEST')
