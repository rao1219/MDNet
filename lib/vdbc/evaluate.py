__author__ = 'raoqi'

from dataset_factory import VDBC
from lib.utils.bbox import bbox_overlaps

class Evaluator(object):
    """class for evaluating tracker."""

    def __init__(self, vdbc, etype='OTE'):
        """Get the VDBC instance.
        Default evaluation method is OTE(one-pass evaluation).
        """
        assert isinstance(vdbc, VDBC), "[Evaluator]Argument vdbc is not the instance of VDBC."
        self._image_list, self._gt_info, self._folder_map = vdbc.get_db(order=True)
        print '[Evaluator]Video set:', self._folder_map
        print '[Evaluator]Number of sets: {}'.format(len(self._folder_map))
        self._etype = etype
        self._ground_truth = None
        self._video_id = 0
        self._video_seq = None
        self._video_name = None
        self._overlaps = {}
        self._init_next_video()

    def set_video(self, vid):
        """Set video set with video id.
        """
        if vid >= len(self._folder_map) or vid < 0:
           print '[Evaluator]Invalid video id.'
           return None
        self._video_id = vid
        self._init_next_video()
        
    def next_video(self):
        """Initialize parameters for evaluating next video sequence.
        """
        self._video_id += 1
        self._init_next_video()

    def _init_next_video(self):
        """Initialize video name, video sequence, overlaps
        list and current frame index.
        """
        self._video_name = self._folder_map[self._video_id]
        self._overlaps[self._video_name] = []
        self._video_seq = self._image_list[self._video_name]
        self._ground_truth = self._gt_info[self._video_name]
        if self._etype == 'OTE':
            self._cur = 0

        print '[Evaluator]Evaluating video {}.'.format(self._video_name)

    def get_results(self):
        return self._overlaps

    def report(self, bbox):
        """tracker calls this function to report the result.
        bbox should be in the form of (x, y, w, h)"""
        gt = self._ground_truth[self._cur - 1]
        # Since result bbox is reported once a time, therefore takes the first one
        overlap = bbox_overlaps([gt], [bbox])[0]
        print '{}: overlap: {}'.format(self._cur-1, overlap)
        self._overlaps[self._video_name].append(overlap)
        return overlap

    def init_frame(self):
        """tracker calls this function to get the necessary frame and ground-truth to initialize.
        """
        if self._etype == 'OTE':
            im = self._video_seq[self._cur]
            gt = self._ground_truth[self._cur]
            self._cur += 1
            return im, gt

    def next_frame(self):
        """tracker calls this function to get the next frame.
        :return im_path if frame is available or None.
        """
        if self._cur < len(self._video_seq):
            im_path = self._video_seq[self._cur] 
            self._cur += 1
            return im_path
        elif self._video_id >= len(self._image_list):
            print '[Evaluator]All video sequences are over.'
            print '[Evaluator]Here is the results:', self._overlaps[self._video_name]
            return None
        else:
            print '[Evaluator]Video sequence is over.'
            return None

    def draw_plots(self):
        """draw the evaluation plot."""
        pass

    def set_etype(self, etype):
        """set the evaluation method."""
        self._etype = etype
    
    def get_ground_truth(self):
        return self._ground_truth[self._cur - 1]

    def get_video_num(self):
        return len(self._image_list)
    
    def get_mAP(self):
        sum = 0.
        for ol in self._overlaps[self._video_name]:
            sum += ol
        return sum / len(self._overlaps[self._video_name])

