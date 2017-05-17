__author__ = 'raoqi'

import _init_paths
import caffe
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from lib.vdbc.dataset_factory import VDBC
from lib.vdbc.evaluate import Evaluator
from lib.vdbc.sample import gaussian_sample, uniform_aspect_sample, uniform_sample, mdnet_sample
from lib.utils.bbox import bbox_reg
from lib.utils.timer import Timer

from lib.data_layer.layer import get_next_mini_batch
from lib.recorder import Recorder

caffe.set_mode_gpu()
caffe.set_device(0)
# get the deploy solver and net with pre-trained caffe model
train = os.path.join('model', 'deploy_solver.prototxt')
test = os.path.join('model', 'deploy_test.prototxt')
weights = os.path.join('model', 'MDNet-otb_iter_6867968.caffemodel')

TEST_PARAMS = [0.1, 0.1, 0.05, 0.7, 0.3]
POS_PARAMS = [0.1, 0.1, 0.05, 0.7, 0.3]
NEG_PARAMS = [2, 2, 5,0.05, 0.3]
INIT_POS_PARAMS = [0.1, 0.1, 0.05, 0.7, 0.5]
INIT_NEG_PARAMS = [1, 1, 0.1, 0.7, 0.5]
INIT_TRAIN_FRAME = 3000

#IMS_PER_FRAME = 40 
IMS_PER_FRAME = 256 + 128
threshold = 0.5

VISUAL = True
STVISUAL = 10
BBOX_REG = True

timer = Timer()
record = Recorder()

    
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
#    time.sleep(1)
    plt.close()

def mean(topInds, scores, samples):
    score = 0.
    targetLoc = [0., 0., 0., 0.]
    for i in range(5):
        ind = topInds[i]
        score += scores[ind]
        box = samples[ind]['box']
        for j in range(4):
            targetLoc[j] += box[j]
    score /= 5
    for i in range(4):
        targetLoc[i] /= 5
    return score, targetLoc

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
    timer.tic()
    db = []
    for ind in seq:
        try:
            samples = frame_samples[ind]
            for i in range(len(samples)):
                db.append({
                'img': samples[i]['img'],
                'samples': [samples[i]]
            })
        except Exception:
            print 'False {} index in sequences finetune'.format(ind)
            pass
    solver.net.layers[0].get_db(db)
    solver.step(1000)
    timer.toc()
    
    print 'Finetune takes {} seconds'.format(timer.diff)

def evaluate(evl, solver, net, sample_num):
    frame_samples = []
    db = []
    
    timer.tic()
    im_path, gt = evl.init_frame()
    im = cv2.imread(im_path)
    
    pos_samples = mdnet_sample(im, gt, INIT_POS_PARAMS, 1000, stype='TEST')
    pos_samples = [sample for sample in pos_samples if sample['label']==1]
    neg_samples = mdnet_sample(im, gt, INIT_NEG_PARAMS, 5000, stype='TEST')
    neg_samples = [sample for sample in neg_samples if sample['label']==0]
    samples = pos_samples + neg_samples
    frame_samples.append(samples)
    for i in range(len(samples)):
        db.append({
            'img': im,
            'samples': [samples[i]]
        })
    solver.net.layers[0].get_db(db)
    solver.step(INIT_TRAIN_FRAME)
    timer.toc()
    print 'Pre-training takes {} seconds.'.format(timer.diff)
    
    if BBOX_REG:
        timer.tic()
        regressor = train_bbox_regression(net, im, gt, samples)
        timer.toc()
        print 'BBox regression training takes {} seconds.'.format(timer.diff)

    long_term = [0]
    short_term = [0]
    finetune_iter_ = 0
#    long_term = []
#    short_term = []
    term = 0
    # Begin testing
    total_timer = Timer()
    total_timer.tic()
    im_path = evl.next_frame()
    while im_path is not None:
        print '--------------------------------------------'
        timer.tic()
        term += 1
        im = cv2.imread(im_path)
        samples = mdnet_sample(im, gt, TEST_PARAMS, sample_num, stype='TEST')
        
        scores = np.zeros(len(samples), dtype=np.float64)
        feats = []
        for i in range(len(samples)):
            db = [{
                'path': im_path,
                'img': im,
                'samples': [samples[i]]
            }]
            blob = get_next_mini_batch(db)
            blob = {'data': blob['data']}
            net.blobs['data'].reshape(*blob['data'].shape)
            out = net.forward(**blob)['cls_prob']
            scores[i] = out[0, 1]
            feats.append(net.blobs['pool3'].data[0])

        ind = np.argmax(scores)
        topInds = np.argsort(scores)[::-1]
        score, targetLoc = mean(topInds, scores, samples)
#         score = scores[topInds[0]]
#         targetLoc = samples[topInds[0]]['box']
        box = np.array(targetLoc)[np.newaxis, :]

        if score > threshold:
            TEST_PARAMS[0] = 0.1
            TEST_PARAMS[1] = 0.1
            
            long_term.append(term)
            short_term.append(term)
            pos_samples = mdnet_sample(im, targetLoc, POS_PARAMS, 100, stype='TEST')
            pos_samples = [sample for sample in pos_samples if sample['label']==1]
            neg_samples = mdnet_sample(im, targetLoc, NEG_PARAMS, 400, stype='TEST')
            pos_samples = [sample for sample in pos_samples if sample['label']==0]
            frame_samples.append(pos_samples + neg_samples)
            
            if len(long_term) >= 100:
                long_term = long_term[-100:]
            if len(short_term) >= 10:
                short_term = short_term[-10:]
            feat = feats[ind]
            feat = feat.reshape((1, feat.size))
            if BBOX_REG:
#                 print 'pre:', box
                box = regressor.predict(feat, box)
#                 print 'after:', box
        timer.toc()
        print 'Prediction takes {} seconds'.format(timer.diff)

        print 'score: {}'.format(score)
        if score < threshold:
            finetune_iter_ +=1
            TEST_PARAMS[0] = 1.1 * TEST_PARAMS[0]
            TEST_PARAMS[1] = 1.1 * TEST_PARAMS[1]
            finetune(solver, frame_samples, short_term)
            frame_samples.append([])
        elif term % 20 == 0:
            finetune_iter_ +=1
            finetune(solver, frame_samples, long_term)
            record._save_json()

        overlap_ = evl.report(box.reshape((4, )))
        gt = box.reshape((4, ))

        record.add_overlap(sample_num,overlap_)
#        if VISUAL or score < threshold or (STVISUAL is not 0 and term % STVISUAL == 0):
#            ground_truth = evl.get_ground_truth()
#            vis_detection(im_path, ground_truth, gt)
        if term % 200 == 0:
            break
        else:
            im_path = evl.next_frame()
    total_timer.toc()
    print 'Total time {} seconds for {} pictures.'.format(total_timer.diff, term)
    print 'mAP: {}.'.format(evl.get_mAP())
    
    record.add_record(sample_num=sample_num,frame_num=term, mAP=evl.get_mAP(), total_time=total_timer.diff, finetune_iter=finetune_iter_)
    record._save_json()


if __name__ == '__main__':

    while IMS_PER_FRAME > 20:
        solver, net = get_solver_net(train, test, weights)

        # get the Evaluator
        dtype = 'VOT'
        dbpath = os.path.join('data', 'VOT')
        gtpath = dbpath

        vdbc = VDBC(dbtype=dtype, dbpath=dbpath, gtpath=gtpath, flush=True)
        evl = Evaluator(vdbc)

        video_num = evl.get_video_num()
        print 'Total video sequences: {}.'.format(video_num)
    #    for i in range(video_num):
    #        evaluate(evl, solver, net)
        
        evl.set_video(19)
        evaluate(evl,solver,net,IMS_PER_FRAME)
        IMS_PER_FRAME -= 5
