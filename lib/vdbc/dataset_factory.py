__author__ = 'stephen'

import cv2
import numpy as np
import re, random
import os, copy, json
from sample import gaussian_sample


class VDBC(object):
    """Experimental class for pre-processing videos."""

    def __init__(self, dbtype, dbpath, gtpath,
                 fimg='image_list.json',
                 fgt='gt_info.json',
                 flush=False):
        """
        Initialize some variables.
        Default setting:
             image list file name is 'image_list.json'
             ground truth information file name is 'gt_info.json'
             flush is False
             ispair is False
        """
        self.__db_types = ['ALOV', 'OTB', 'VOT']
        self.__db_path = dbpath
        self.__gt_path = gtpath
        self.__fimg = fimg
        self.__fgt = fgt
        assert dbtype in self.__db_types, "Not known database. {}".format(dbtype)
        assert os.path.isdir(dbpath), "Invalid database path. {}".format(dbpath)
        assert os.path.isdir(gtpath), "Invalid ground-truth path. {}".format(gtpath)

        # Initialize image_list
        self.__image_list = None
        self.__gt_info = None

        if dbtype == 'ALOV':
            self.roidb_ALOV(flush)
        elif dbtype == 'OTB':
            self.roidb_OTB(flush)
        elif dbtype == 'VOT':
            self.roidb_VOT(flush)

        print 'VDBC instance built.'

    def get_db(self, order=False):
        """Get the database information. foldr map is ordered if order is true.
        """
        assert self.__gt_info is not None, "Ground-truth not loaded."
        assert self.__image_list is not None, "Image list is not loaded."

        if order is True:
            self.__folder_map.sort()
        return self.__image_list, self.__gt_info, self.__folder_map

    def roidb_ALOV(self, flush):
        """
        Get the region of interest on ALOV database.
        ALOV database last-level folder architecture should be in the form:
            folder1(e.g. 01-Light)
                -> folder1(e.g. 01-Light_Video00001)
                    -> images(e.g. 00000001.jpg ...)
        """

        # First check whether there are pre-processed json file
        # if so, then load the files, otherwise process the database.
        if os.path.exists(self.__fimg) and os.path.exists(self.__fgt) and not flush:
            with open(self.__fimg, "r") as fimg:
                self.__image_list = json.load(fimg)
                print 'Successfully load json file {}.'.format(self.__fimg)
            with open(self.__fgt, "r") as fgt:
                self.__gt_info = json.load(fgt)
                print 'Successfully load json file {}.'.format(self.__fgt)
        else:
            self.__image_list = self.fget_last_level_list(self.__db_path)
            gt_txt = self.fget_last_level_list(self.__gt_path)

            # get information from the ground-truth text files
            self.__gt_info = {}
            for folder in gt_txt:
                for text in gt_txt[folder]:
                    video = text.split(os.sep)[-1].split('.')[0]
                    self.__gt_info[video] = []
                    pre_num = 1
                    cur_box = [None]
                    for line in open(text).readlines():
                        line = line.split()
                        # form of each line in ALOV is "number x1 y1 x2 y2 x3 y3 x4 y4"
                        curr_num = int(line[0])
                        self.__gt_info[video].extend(cur_box * (curr_num - pre_num))
                        pre_num = curr_num
                        cur_box = [self.rect_box(line[1:])]
                    # Get the number of images and extend the boxes
                    imgs_num = len(self.__image_list[video]) + 1
                    self.__gt_info[video].extend(cur_box * (imgs_num - pre_num))

            for video in self.__image_list:
                self.__image_list[video].sort()
            self._save_json_text(self.__image_list, fname=self.__fimg)
            self._save_json_text(self.__gt_info, fname=self.__fgt)

        self.__num_videos = len(self.__image_list)
        self.__folder_map = [folder for folder in self.__image_list]

    def roidb_OTB(self, flush):
        """
        Get the region of interest on OTB database.
        OTB folder architecture should be in the form:
        OTB
        -> folder1(e.g. Basketball)
            -> folder2(img)
                -> images(e.g. 0001.jpg)
            -> ground-truth text(groundtruth_rect.txt)
        """

        # Get all the video sets
        video_list = [item for item in os.listdir(self.__db_path)]

        # First check whether there are pre-processed json file
        # if so, then load the files, otherwise process the database.
        if os.path.exists(self.__fimg) and os.path.exists(self.__fgt) and not flush:
            with open(self.__fimg, "r") as fimg:
                self.__image_list = json.load(fimg)
                print 'Successfully load json file {}.'.format(self.__fimg)
            with open(self.__fgt, "r") as fgt:
                self.__gt_info = json.load(fgt)
                print 'Successfully load json file {}.'.format(self.__fgt)
        else:
            vf = 'img'
            gtf = 'groundtruth_rect.txt'

            self.__image_list = {}
            self.__gt_info = {}
            for folder in video_list:
                _video = self.__db_path + os.sep + folder
                if not os.path.isdir(_video):
                    continue
                # __image_list of OTB is in the form of "folder : [file1_path, file2_path, ...]"
                self.__image_list[folder] = [_video + os.sep + vf + os.sep + image for image in
                                             os.listdir(self.__db_path + os.sep + folder + os.sep + vf)]
                # Read the ground-truth text file
                with open(self.__db_path + os.sep + folder + os.sep + gtf, "r") as f:
                    gt_list = [re.findall(r'[0-9]+', line) for line in f.readlines()]
                # form of each line in OTB is "x y w h"
                self.__gt_info[folder] = [(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                                          for box in gt_list]

            for video in self.__image_list:
                self.__image_list[video].sort()
            self._save_json_text(self.__image_list, fname=self.__fimg)
            self._save_json_text(self.__gt_info, fname=self.__fgt)

        self.__num_videos = len(self.__image_list)
        self.__folder_map = video_list

    def roidb_VOT(self, flush):
        """
        Get the region of interest on VOT database.
        VOT folder architecture should be in the form:
        -> list.txt
        -> folders(e.g. Basketball)
            -> images(e.g. 0001.jpg)
            -> ground-truth text(groundtruth_rect.txt)
        """
        video_list = []
        with open(self.__db_path + os.sep + 'list.txt', "r") as db:
            for folder in db.readlines():
                video_list.append(folder)

        # First check whether there are pre-processed json file
        # if so, then load the files, otherwise process the database.
        if os.path.exists(self.__fimg) and os.path.exists(self.__fgt) and not flush:
            with open(self.__fimg, "r") as fimg:
                self.__image_list = json.load(fimg)
                print 'Successfully load json file {}.'.format(self.__fimg)
            with open(self.__fgt, "r") as fgt:
                self.__gt_info = json.load(fgt)
                print 'Successfully load json file {}.'.format(self.__fgt)
        else:
            gtf = 'groundtruth.txt'

            self.__image_list = {}
            self.__gt_info = {}
            for folder in video_list:
                _video = self.__db_path + os.sep + folder
                if not os.path.isdir(_video):
                    continue
                # __image_list of VOT is in the form of "folder : [file1_path, file2_path, ...]"
                self.__image_list[folder] = [_video + os.sep + image for image in
                                             os.listdir(self.__db_path + os.sep + folder)
                                             if os.path.splitext(image)[1] == '.jpg']
                # Read the ground-truth text file
                raw_data = np.loadtxt(_video + os.sep + gtf, delimiter=',',
                                      usecols=(0, 1, 2, 3, 4, 5, 6, 7), dtype=np.float)
                # data is a list of (maxx, minx, maxy, miny)
                data = [(max((box[0], box[2], box[4], box[6])),
                         min((box[0], box[2], box[4], box[6])),
                         max((box[1], box[3], box[5], box[7])),
                         min((box[1], box[3], box[5], box[7]))) for box in raw_data]
                # form of each line in VOT is "x y w h"
                self.__gt_info[folder] = [(box[1], box[3], box[0] - box[1], box[2] - box[3])
                                          for box in data]

            for video in self.__image_list:
                self.__image_list[video].sort()
            self._save_json_text(self.__image_list, fname=self.__fimg)
            self._save_json_text(self.__gt_info, fname=self.__fgt)

        self.__num_videos = len(self.__image_list)
        self.__folder_map = video_list

    def rect_box(self, box, dtype=float):
        """Transform [x1, y1, x2, y2, x3, y3, x4, y4] to [x, y, w, h]"""
        bbox = [dtype(x) for x in box]
        x_axis, y_axis = bbox[::2], bbox[1::2]
        x_axis.sort(), y_axis.sort()
        return (x_axis[0],
                y_axis[0],
                x_axis[-1] - x_axis[0],
                y_axis[-1] - y_axis[0])

    def fget_last_level_list(self, path):
        """Traverse the directory and get all the last-level file paths."""
        file_list = {}
        path_list = [path]  # path_list is a level-list
        while len(path_list) > 0:  # path_list will be set as empty if it reaches the last level
            path_num = len(path_list)
            tmp = copy.deepcopy(path_list)
            for folder in tmp:
                if os.path.isdir(folder):
                    lists = [folder + os.sep + item for item in os.listdir(folder)]
                    if os.path.isdir(lists[0]):
                        path_list.extend(lists)
                    else:
                        # file_list is a dict in the form of {"folder_name":[file1_path, file2_path ...]}
                        # lists.sort(reverse=True)
                        file_list[re.split(r'\\|/', folder)[-1]] = lists
            path_list = path_list[path_num:]
        return file_list

    def build_data(self, params, num, dtype='GAUSSIAN'):
        """
        Build single-image.
        params: parameters about generating samples
        Number of the samples is 'num'.
        """
        gt = None
        im = None
        samples = None
        frame = None
        while gt is None:
            rd_video = random.randint(0, self.__num_videos - 1)
            video_img = self.__image_list[self.__folder_map[rd_video]]
            rd_img = random.randint(0, len(video_img) - 1)

            frame = video_img[rd_img]
            im_index = self._get_im_index(frame)
            gt = self.__gt_info[self.__folder_map[rd_video]][im_index]
            if gt is None:
                continue

            print self.__folder_map[rd_video], im_index, gt
            im = cv2.imread(frame)
            if dtype == 'GAUSSIAN':
                samples = gaussian_sample(im=im, bbox=gt, params=params, num=num)

        # build data
        # samples is a list of boxes{'img':img, 'box': (x, y ,w, h), 'label':label}
        data = {
            'path': frame,
            'img': im,
            'gt': gt,
            'samples': samples
        }

        return data

    def build_pair_data(self, params=None, im_per_pair_frame=256):
        """
        Build pair data according to image list in the video set.
        For each sampling strategy, number of each pair-frame sample is cfg.IMG_PAIR_PER_FRAME.
        """

        gt1 = gt2 = None
        frame1 = frame2 = None
        while gt1 is None or gt2 is None:
            rd_video = random.randint(0, self.__num_videos - 1)
            video_imgs = self.__image_list[self.__folder_map[rd_video]]

            # randomly choose two frames
            rd_img1 = random.randint(0, len(video_imgs) - 1)
            rd_img2 = random.randint(0, len(video_imgs) - 1)
            frame1 = video_imgs[rd_img1]
            frame2 = video_imgs[rd_img2]

            # get the ground truth information
            gt1 = self.__gt_info[self.__folder_map[rd_video]][rd_img1]
            gt2 = self.__gt_info[self.__folder_map[rd_video]][rd_img2]

        img1 = cv2.imread(frame1)
        img2 = cv2.imread(frame2)
        # draw gaussian samples
        rois = gaussian_sample(img2, gt2, params, im_per_pair_frame)
        samples = [{
                       'box1': gt1,
                       'box2': bbox['box'],
                       'label': bbox['label']}
                   for bbox in rois]

        pair_data = {
            'img1': img1,
            'img2': img2,
            'gt1': gt1,
            'gt2': gt2,
            'samples': samples
        }

        return pair_data

    def _get_im_index(self, im_path):
        """Get index of the image in the ground-truth text.
        :param image path
        """
        im_name = re.split(r'\\|/', im_path)[-1]
        id = im_name.split('.')[0]
        return int(id) - 1

    def frame_index(self, frame):
        """Get the index of the frame according to the frame file name."""
        basename = re.split(r'\\|/', frame)[-1]
        return int(basename.split('.')[0])

    def _save_json_text(self, info, fname='data.json'):
        with open(fname, "w") as f:
            f.write(json.dumps(info))
        print 'save {} successfully.'.format(fname)

    def _load_json_text(self, fname='data.json'):
        with open(fname, "r") as f:
            data = json.load(f)
        return data
