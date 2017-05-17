__author__ = 'raoqi'

import caffe
import yaml, copy
import numpy as np
from lib.vdbc.dataset_factory import VDBC
from lib.utils.blob import im_list_to_blob, prep_im_for_blob
from layer_config import cfg


def get_image_blob(db, pixel_means):
    processed_ims = []

    for data in db:
        img = data['img']
        img = img.astype(np.float32, copy=False)
        img -= pixel_means

        for sample in data['samples']:
            box = list(sample['box'])
            for i in range(len(box)):
                box[i] = int(box[i])
            im = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            im = prep_im_for_blob(im,
                                  cfg.TRAIN.INPUT_SIZE,
                                  cfg.TRAIN.INPUT_SIZE)
            processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)

    return blob


def get_next_mini_batch(db, pixel_means=None):
    num_images = len(db)

    if pixel_means is None:
        # It is a default pixel mean from py-faster-rcnn even though it may be not exact
        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

    im_blob = get_image_blob(db, pixel_means)
    labels_blob = np.zeros(0, dtype=np.float32)

    blobs = {
        'data': im_blob
    }

    for im_i in range(num_images):
        labels = [sample['label'] for sample in db[im_i]['samples']]
        labels_blob = np.hstack((labels_blob, np.array(labels)))
    blobs['label'] = labels_blob

    return blobs


def get_next_mul_label_batch(db, num_class, pixel_means=None):
    num_images = len(db)

    if pixel_means is None:
        # It is a default pixel mean from py-faster-rcnn even though it may be not exact
        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

    im_blob = get_image_blob(db, pixel_means)
    labels_blob = np.zeros((1, num_class), dtype=np.float32)

    blobs = {
        'data': im_blob
    }

    for im_i in range(num_images):
        label = [sample['label'] for sample in db[im_i]['samples']]
        assert len(label) == 1, "[DataLayer.Batch]Only support single batch."
        label = label[0]
        if label >= 0:
            labels_blob[0, label] = 1

    blobs['label'] = labels_blob

    return blobs


class DataLayer(caffe.Layer):
    """MDNet video data layer for training."""

    def get_db(self, db):
        self._db = db
        self._vdbc = None # unset VDBC instance
        self._cur = 0
        self._perm = np.random.permutation(np.arange(len(self._db)))
        print '[DataLayer] Get the database.'

    def get_VDBC(self, vdbc):
        """Get VDBC instance."""
        assert isinstance(vdbc, VDBC), "It is not a VDBC instance."
        self._vdbc = vdbc
        self._build_new_minidb()

    def _build_new_minidb(self):
        """Build new mini-database."""
        self._db = self._vdbc.build_data_in_list_order(cfg.TRAIN.PARAMS,
                                                       cfg.TRAIN.IMS_PER_BATCH)
        print '[DataLayer] New mini-database built.'
        self._cur = 0
        self._perm = np.random.permutation(np.arange(len(self._db)))

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._batch_size > len(self._db):
            if self._vdbc is not None:
                self._build_new_minidb()
            else:
                self._cur = 0
                self._perm = np.random.permutation(np.arange(len(self._db)))

        db_inds = self._perm[self._cur:self._cur + self._batch_size]
        self._cur += self._batch_size
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next mini-batch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minidb = [self._db[i] for i in db_inds]
        return get_next_mul_label_batch(minidb, self._num_class) \
            if self._num_class > 2 else get_next_mini_batch(minidb)

    def setup(self, bottom, top):
        """Setup the DataLayer."""

        # parse the layer parameter string
        layer_params = yaml.load(self.param_str)

        self._batch_size = 1 # batch size is always 1
        self._num_class = layer_params['num_class']
        print '[DataLayer]Number of classes: {}.'.format(self._num_class)

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(self._batch_size, 3,
                         cfg.TRAIN.INPUT_SIZE,
                         cfg.TRAIN.INPUT_SIZE)
        self._name_to_top_map['data'] = idx

        idx += 1
        if self._num_class>2:
            top[idx].reshape(1, self._num_class)
        else:
            top[idx].reshape(1)
        self._name_to_top_map['label'] = idx

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
