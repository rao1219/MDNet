__author__ = 'stephen'

import caffe
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from lib.vdbc.dataset_factory import VDBC
from lib.vdbc.evaluate import Evaluator
from lib.vdbc.sample import gaussian_sample
from lib.utils.bbox import bbox_reg

from lib.data_layer.layer import get_next_mini_batch

caffe.set_mode_gpu()
caffe.set_device(0)
# get the deploy solver and net with pre-trained caffe model
train = os.path.join('model', 'deploy_solver.prototxt')
test = os.path.join('model', 'deploy_test.prototxt')
weights = os.path.join('model', 'MDNet_10epoch_pad.caffemodel')

PARAMS = (0.3, 0.3, 0.05, 0.7, 0.3)
INIT_TRAIN_FRAME = 5500
IMS_PER_FRAME = 256

threshold = 0.5

VISUAL = True
BBOX_REG = True


def vis_detection(im_path, gt, box):
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


def get_solver_net(train, test, weights):
    solver = caffe.SGDSolver(train)
    solver.net.copy_from(weights)

    net = caffe.Net(test, caffe.TEST)
    net.share_with(solver.net)

    return solver, net


def train_bbox_regression(net, im, gt, frame_samples):
    bboxes = []
    X = None
    for sample in frame_samples:
        if sample['label'] == 1:
            db = [{
                'img': im,
                'samples': [sample]
            }]
            blob = get_next_mini_batch(db)
            blob = {'data': blob['data'].astype(np.float32, copy=True)}
            net.blobs['data'].reshape(*blob['data'].shape)
            net.forward(data=blob['data'])
            x = net.blobs['pool3'].data[0]
            x = x.reshape((1, x.size))
            if X is None:
                X = x
            else:
                X = np.vstack((X, x))

            bboxes.append({
                'box': sample['box'],
                'label': 1,
                'overlap': sample['overlap']
            })

    regressor = bbox_reg()
    regressor.train(X, bboxes, gt)
    return regressor


def finetune(solver, frame_samples, seq):
    db = []
    for ind in seq:
        db.extend(frame_samples[ind])
    finetune_db = []
    for i in range(len(db)):
        finetune_db.append({
            'img': db[i]['img'],
            'samples': [db[i]]
        })
    solver.net.layers[0].get_db(db)
    solver.step(2000)


def evaluate(evl, solver, net):
    samples = []
    db = []

    im_path, gt = evl.init_frame()
    im = cv2.imread(im_path)
    frame_samples = gaussian_sample(im, gt, PARAMS, INIT_TRAIN_FRAME)
    for i in range(len(frame_samples)):
        db.append({
            'path': im_path,
            'img': im,
            'samples': [samples[i]]
        })
    solver.net.layers[0].get_db(db)
    solver.step(5000)
    if BBOX_REG:
        regressor = train_bbox_regression(net, im, gt, frame_samples)

    long_term = [0]
    short_term = [0]
    term = 0
    # Begin testing
    im_path = evl.next_frame()
    while im_path is not None:
        term += 1
        im = cv2.imread(im_path)
        samples = gaussian_sample(im, gt, PARAMS, IMS_PER_FRAME)
        frame_samples.append(samples)

        scores = np.zeros(IMS_PER_FRAME, dtype=np.float64)
        feats = []
        for i in range(IMS_PER_FRAME):
            db = [{
                'path': im_path,
                'img': im,
                'samples': [samples[i]]
            }]
            blob = get_next_mini_batch(db)
            blob = {'data': blob['data']}
            net.blobs['data'].reshape(*blob['data'].shape)
            out = net.forward(**blob)['cls_prob']
            scores[i] = out[1]
            feats.append(net.blobs['pool3'].data[0])
        ind = np.argmax(scores)
        score = scores[ind]
        box = np.array(samples[ind]['box']).reshape((1, 4))

        if score > threshold:
            long_term.append(term)
            short_term.append(term)
            if len(long_term) >= 100:
                long_term = long_term[-100:]
            if len(short_term) >= 20:
                short_term = short_term[-20:]
            feat = feats[ind]
            feat = feat.reshape((1, feat.size))
            if BBOX_REG:
                box = regressor.predict(feat, box)

        if score < threshold:
            finetune(solver, frame_samples, short_term)
        elif term % 10 == 0:
            finetune(solver, frame_samples, long_term)

        evl.report(box.reshape((4, )))
        gt = box.reshape((4, ))

        if VISUAL:
            ground_truth = evl.get_ground_truth()
            vis_detection(im_path, ground_truth, gt)

        im_path = evl.next_frame()

if __name__ == '__main__':
    solver, net = get_solver_net(train, test, weights)

    # get the Evaluator
    dtype = 'VOT'
    dbpath = os.path.join('data', 'vot2014')
    gtpath = dbpath

    vdbc = VDBC(dbtype=dtype, dbpath=dbpath, gtpath=gtpath, flush=True)
    evl = Evaluator(vdbc)

    video_num = evl.get_video_num()
    print 'Total video sequences: {}.'.format(video_num)
    for i in range(video_num):
        evaluate(evl, solver, net)
