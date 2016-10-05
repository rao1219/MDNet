__author__ = 'stephen'

import numpy as np
import math


def bbox_overlaps(gt_boxes, query_boxes, gt_fix=True):
    """
     Calculate the overlaps between ground-truth boxes and query boxes.
     boxes are in the form of (x, y, w, h).
     :return list of corresponding overlaps
    """
    overlaps = []

    if gt_fix:
        assert len(gt_boxes) == 1, "[bbox_overlaps]There should be only one ground-truth box."
        gt_box = gt_boxes[0]

        box_area = (gt_box[2] * gt_box[3])
        for query_box in query_boxes:
            iw = (
                min(gt_box[0] + gt_box[2], query_box[0] + query_box[2]) -
                max(gt_box[0], query_box[0])
            )
            if iw > 0:
                ih = (
                    min(gt_box[1] + gt_box[3], query_box[1] + query_box[3]) -
                    max(gt_box[1], query_box[1])
                )
                if ih > 0:
                    ua = float(
                        box_area +
                        query_box[2] * query_box[3] -
                        iw * ih
                    )
                    overlaps.append(iw * ih / ua)
                else:
                    overlaps.append(0)
            else:
                overlaps.append(0)

    # TODO: unfixed overlaps
    return overlaps


def add_box_clr(box, clr):
    """Add color element to the box.
    (x, y, w, h) to (x, y, w, h, clr)
    """
    assert len(box) == 4, "[add_box_clr]Invalid box."
    return box[0], box[1], box[2], box[3], clr


class bbox_reg(object):

    def __init__(self, min_overlap=0.6, ld=1000, robust=0):
        self._min_overlap = min_overlap
        self._ld = ld
        self._robust = robust

        self.mu = None
        self.T = None
        self.T_inv = None
        self.Beta = None

    def train(self, X, bboxes, gt):
        """
        This class is actually after MDNet's train_bbox_regressor.m
        bboxes: list of boxes
            {
                'box'(x, y, w, h),
                'label': label,
                'overlap':overlap
            }
        gt: four-tuple(x, y, w, h)
        """
        # Get positive examples
        Y, O = self._get_examples(bboxes, gt)

        idx = np.where(O > self._min_overlap)[0]
        X = X[idx]
        Y = Y[idx]
        # add bias
        X = np.column_stack((X, np.ones(X.shape[0], dtype=np.float32)))

        # Center and decorrelate targets
        mu = np.mean(Y)
        Y = Y - mu
        S = np.dot(Y.T, Y) / Y.shape[0]
        D, V = np.linalg.eig(S)
        T = V.T.dot(V.dot(np.diag(1 / np.sqrt(D + 0.001))))
        T_inv = V.T.dot(V.dot(np.diag(np.sqrt(D + 0.001))))
        Y = np.dot(Y, T)

        self.Beta = [self._solve_robust(X, Y[:, 0], self._ld, self._robust),
                     self._solve_robust(X, Y[:, 1], self._ld, self._robust),
                     self._solve_robust(X, Y[:, 2], self._ld, self._robust),
                     self._solve_robust(X, Y[:, 3], self._ld, self._robust)]

        self.mu = mu
        self.T = T
        self.T_inv = T_inv

    def predict(self, feat, ex_boxes):
        end = self.Beta.shape[1]
        # Predict regression targets
        Y = feat * self.Beta[:end-1, :] + self.Beta[end-1, :]
        # Invert whitening transformation
        Y = Y * self.T_inv + self.mu

        # Read out predictions
        dst_ctr_x = Y[:, 0]
        dst_ctr_y = Y[:, 1]
        dst_scl_x = Y[:, 2]
        dst_scl_y = Y[:, 3]

        src_w = ex_boxes[:, 2]
        src_h = ex_boxes[:, 3]
        src_ctr_x = ex_boxes[:, 0] + 0.5 * src_w
        src_ctr_y = ex_boxes[:, 1] + 0.5 * src_h

        pred_ctr_x = (dst_ctr_x * src_w) + src_ctr_x
        pred_ctr_y = (dst_ctr_y * src_h) + src_ctr_y
        pred_w = np.exp(dst_scl_x) * src_w
        pred_h = np.exp(dst_scl_y) * src_h
        pred_boxes = np.zeros((Y.shape[0], 4))
        pred_boxes[:, 0] = (pred_ctr_x - 0.5 * pred_w)[:, 0]
        pred_boxes[:, 1] = (pred_ctr_y - 0.5 * pred_h)[:, 0]
        pred_boxes[:, 2] = pred_w[:, 0]
        pred_boxes[:, 3] = pred_h[:, 0]

        return pred_boxes

    def _get_examples(self, bboxes, gt):

        Y = np.zeros((0, 4), dtype=np.float32)
        O = np.zeros(0, dtype=np.float32)

        gt_w = 1. * gt[2]
        gt_h = 1. * gt[3]
        gt_ctr_x = gt[0] + 0.5 * gt_w
        gt_ctr_y = gt[1] + 0.5 * gt_h

        for bbox in bboxes:
            ex_box = bbox['box']
            ex_overlap = bbox['overlap']

            src_w = ex_box[2]
            src_h = ex_box[3]
            src_ctr_x = ex_box[0] + 0.5 * src_w
            src_ctr_y = ex_box[1] + 0.5 * src_h

            dst_ctr_x = (gt_ctr_x - src_ctr_x) * 1. / src_w
            dst_ctr_y = (gt_ctr_y - src_ctr_y) * 1. / src_h
            dst_scl_w = math.log(gt_w / src_w)
            dst_scl_h = math.log(gt_h / src_h)

            arr = [dst_ctr_x, dst_ctr_y, dst_scl_w, dst_scl_h]
            Y = np.vstack((Y, np.array(arr, dtype=np.float32)))
            O = np.hstack((O, np.array(ex_overlap, dtype=np.float32)))

        return Y, O

    def _solve_robust(self, A, y, ld, qtile):
        x, losses = self._solve(A, y, ld)
        if qtile > 0:
            pass
        return x, losses

    def _solve(self, A, y, ld):
        M = A.T.dot(A)
        E = np.eye(A.shape[1])
        E *= ld
        M += E
        R = np.linalg.cholesky(M)
        z = R.T / A.T.dot(y)
        x = R / z
        losses = 0.5 * np.square(A.dot(x) - y)
        return x, losses

if __name__ == '__main__':
    import cv2, os
    import pickle as pkl
    from lib.vdbc.sample import gaussian_sample

    PARAMS = (0.3, 0.3, 0.05, 0.7, 0.3)
    conv_data = pkl.load(open('conv_data.pkl', 'rb'))

    data = conv_data[0]
    num_feat = len(data['boxes'])
    X = None
    bboxes = []
    for i in range(num_feat):
        x = data['data'][i]
        x = x.reshape((1, x.size))
        if X is None:
            X = x
        else:
            X = np.vstack((X, x))
        box = data['boxes'][i]
        overlap = data['overlap'][i]
        bboxes.append({
            'box': box,
            'label': 1,
            'overlap': overlap
        })

    reg = bbox_reg()
    reg.train(X, bboxes, data['gt'])
