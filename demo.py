__author__ = 'stephen'

import caffe
import cv2, os
import matplotlib.pyplot as plt
from lib.vdbc.dataset_factory import VDBC
from lib.vdbc.evaluate import Evaluator
from lib.vdbc.sample import gaussian_sample

from lib.data_layer.layer import get_next_mini_batch

PARAMS = (0.3, 0.3, 0.05, 0.7, 0.3)


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


def evaluate(evl, solver, net):
    im_path, gt = evl.init_frame()

    def initialize():
        im = cv2.imread(im_path)
        db = []
        for i in range(30):
            samples = gaussian_sample(im, gt, PARAMS, 64)
            db.append({
            'path': im_path,
            'img': im,
            'gt': gt,
            'samples': samples
            })
        solver.net.layers[0].get_db(db)
        solver.step(30)

    scores = []

    # Initialize the net with the first frame
    initialize()

    for i in range(1):
        im_path = evl.next_frame()
        im = cv2.imread(im_path)
        samples = gaussian_sample(im, gt, PARAMS, 64)
        db = [{
            'path': im_path,
            'img': im,
            'gt': gt,
            'samples': samples
        }]
        blob = get_next_mini_batch(db)
        blob = {'data': blob['data']}

        net.blobs['data'].reshape(*blob['data'].shape)

        out = net.forward(**blob)['cls_prob']
        print out


if __name__ == '__main__':
    caffe.set_mode_gpu()

    # get the deploy solver and net with pre-trained caffe model
    train = os.path.join('model', 'deploy_solver.prototxt')
    test = os.path.join('model', 'deploy_test.prototxt')
    weights = os.path.join('model', 'MDNet_iter_80000.caffemodel')

    solver, net = get_solver_net(train, test, weights)

    # get the Evaluator
    dtype = 'VOT'
    dbpath = os.path.join('data', 'vot2014')
    gtpath = dbpath

    vdbc = VDBC(dbtype=dtype, dbpath=dbpath, gtpath=gtpath, flush=True)

    evl = Evaluator(vdbc)
    evl.set_video(1)

    evaluate(evl, solver, net)
