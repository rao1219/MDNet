#!/usr/bin/env python
# coding=utf-8
__author__ = 'raoqi'

import _init_paths
import os
import cv2
import matplotlib.pyplot as plt
from lib.vdbc.dataset_factory import VDBC
from lib.vdbc.evaluate import Evaluator
from lib.vdbc.sample import *


PARAMS = (2, 2, 0.05, 0.7, 0.3)

def vis_detection(im_path, gt, boxes):
    im = cv2.imread(im_path)[:, :, (2, 1, 0)]
    plt.cla()
    plt.imshow(im)
    # add ground-truth box
    plt.gca().add_patch(
        plt.Rectangle(
            (gt[0], gt[1]),
            gt[2], gt[3],
            fill=False,
            edgecolor='red',
            linewidth=1.5
        )
    )
    # add detection box
    for box in boxes:
        plt.gca().add_patch(
            plt.Rectangle(
                (box[0], box[1]),
                box[2], box[3],
                fill=False,
                edgecolor='blue',
                linewidth=1.5
            )
        )

    plt.show()

if __name__ == '__main__':
    IM_PER_FRAME = 256
    dtype = 'VOT'
    dbpath = os.path.join('data','VOT')
    gtpath = dbpath

    vdbc = VDBC(dbtype=dtype, dbpath=dbpath, gtpath=gtpath, flush=True)
    evl = Evaluator(vdbc)

    evl.set_video(3)
    im_path, gt = evl.init_frame()
    im = cv2.imread(im_path)
    frame_samples = mdnet_sample(im ,gt, PARAMS, IM_PER_FRAME, 'TEST')
    #frame_samples = uniform_sample(im ,gt, PARAMS, IM_PER_FRAME, 'TEST')

    bboxes = [sample['box'] for sample in frame_samples ]
    vis_detection(im_path,gt,bboxes)

    #print type(bboxes)
