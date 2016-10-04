__author__ = 'stephen'

####################################
# Author: Zilin Zhang
# Organization: Zhejiang University
# City: Hangzhou
# Description: The file contains
# multiple sample methods
####################################

import cv2
import random as rd
from lib.utils.bbox import bbox_overlaps


def gaussian_sample(im, bbox, params, num):
    """Generate gaussian samples based on bbox
    :arg
    im: cv2's image
    bbox: ground-truth box(x, y, w, h)
    params: five-tuple(width, height, scale, pos_threshold, neg_threshold) of gaussian parameters
    num: number of samples

    :return
    bboxes: list of boxes
            {
                'img' :img,
                'box'(x, y, w, h),
                'label': label
            }
    """
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

    bboxes = []
    cur_id = 0
    while cur_id < num:
        # new box parameters
        offsetx = rd.gauss(0, params[0] * bbox[2])
        offsety = rd.gauss(0, params[1] * bbox[3])
        scalex = rd.gauss(1, params[2])
        scaley = rd.gauss(1, params[2])
        # new box half width and half height
        hw = bbox[2] * scalex / 2
        hh = bbox[3] * scaley / 2
        # box is in the form of (x1, y1, x2, y2)
        box = (
            max(0, centerx + offsetx - hw),
            max(0, centery + offsety - hh),
            min(im_w, centerx + offsetx + hw),
            min(im_h, centery + offsety + hh)
        )

        if box[0] > box[2] or box[1] > box[3]:
            continue

        # transform to (x, y, w, h)
        sample = (box[0], box[1], box[2] - box[0], box[3] - box[1])
        # since there is only one query box, then take the first one in the overlaps
        overlap = bbox_overlaps([bbox], [sample])[0]
        if overlap > params[3]:
            bboxes.append({
                'img': im,
                'box': sample,
                'label': 1})
        elif overlap < params[4]:
            bboxes.append({
                'img': im,
                'box': sample,
                'label': 0})
        else:
            continue
        cur_id += 1
    return bboxes


if __name__ == '__main__':
    """Experiment code for testing.
    """
    from lib.utils.image import im_bbox_show
    from dataset_factory import VDBC
    vdbc = VDBC(dbtype='ALOV',
                gtpath='C:\\Users\\user\\Desktop\\alov300++_frames\\gt',
                dbpath='C:\\Users\\user\\Desktop\\alov300++_frames\\video')


    im_list, gts, fd_map = vdbc.get_db()
    im_path = im_list[fd_map[2]][11]
    im = cv2.imread(im_path)
    gt = gts[fd_map[2]][11]

    bboxes = gaussian_sample(im, gt, (0.2, 0.2, 0.05, 0.7), 12)

    boxes = [(gt[0], gt[1], gt[2], gt[3], 'blue')]
    pos = 0
    for sample in bboxes:
        box = sample['box']
        if sample['label'] == 1:
            boxes.append((box[0], box[1], box[2], box[3], 'red'))
            pos += 1
        else:
            boxes.append((box[0], box[1], box[2], box[3], 'green'))
    print "pos: ", pos
    im_bbox_show(im_path, boxes, linewidth=1.5)


