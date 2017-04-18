#!/usr/bin/env python
# coding=utf-8
__author__ = 'raoqi'
import json
import os
import matplotlib.pyplot as plt
import numpy as np

class Recorder(object):
    """Recorder for timer output"""
    
    def __init__(self,info=None):
        if not info is None:
            self.__timer_info = info    
        else:
            self.__timer_info = []  
        self.__overlap_info = []

    def get_data(self):
        """overlap, timer"""
        return self.__overlap_info, self.__timer_info

    def add_record(self,sample_num,frame_num, mAP, total_time, finetune_iter):
        record = {}
        record['sample_num'] = sample_num
        record['frame_num'] = frame_num
        record['mAP'] = mAP
        record['total_time'] = total_time
        record['finetune_iter'] = finetune_iter

        self.__timer_info.append(record)
        
    def add_overlap(self, sample_num, _overlap):
        overlap = {}
        overlap['sample_num'] = sample_num
        overlap['lap'] = _overlap
        self.__overlap_info.append(overlap)

    def _save_json(self, fname="res.json", foverlap="overlap.json"):
        if not len(self.__timer_info) == 0: 
            with open(fname,'w') as f:
                f.write(json.dumps(self.__timer_info))
        
        if not len(self.__overlap_info) == 0: 
            with open(foverlap, 'w') as f:
                f.write(json.dumps(self.__overlap_info))
            
        print 'save result data successfully.' 

    def _load_json(self, fname="res.json", foverlap="overlap.json"):
        if os.path.exists(fname):
            with open(fname,'r') as f:
                self.__timer_info = json.load(f)

        if os.path.exists(foverlap):
            with open(foverlap,'r') as f:
                self.__overlap_info = json.load(f)
            
            
if __name__ == '__main__':
    record = Recorder()
    record._load_json()
    
    lap, ii = record.get_data()
    total = len(lap)
    lap = [lp['lap'] for lp in lap if lp['lap'] >0.5]
#    #laplist = [lp for lp in lap if lp['sample_num'] <384]
#    #print laplist
#    print 'length:' , len(lap)
#    print 'acc:' , len(lap)*1.0/total 
    
    sample =  [i['sample_num'] for i in ii]
    mAP = [i['mAP'] for i in ii]
    time = [i['total_time'] for i in ii]
    iter_ = [i['finetune_iter'] for i in ii]

    print len(ii)
    
    f, addr = plt.subplots(2,2)
    addr[0,0].set_title('sample number')
    addr[0,0].plot(sample)

    addr[0,1].set_title('iter')
    addr[0,1].plot(sample,iter_)

    addr[1,0].set_title('mAP')
    addr[1,0].plot(sample,mAP)
   
    addr[1,1].set_title('time')
    addr[1,1].plot(sample,time)
    plt.show()
