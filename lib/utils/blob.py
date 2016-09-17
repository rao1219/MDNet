# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, width, height):
    """Scale an image for use in a blob."""
    im_shape = im.shape
    im_size_x = im_shape[1]
    im_size_y = im_shape[0]
    im_scale_x = float(width) / float(im_size_x)
    im_scale_y = float(height) / float(im_size_y)
    # Resize the image
    im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
                    interpolation=cv2.INTER_LINEAR)

    return im
