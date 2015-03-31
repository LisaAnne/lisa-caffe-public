#!/usr/bin/env python

import sys
sys.path.append('/home/lisaanne/caffe-forward-backward/python')
import caffe
from process_hdf5_general import *
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import time
import pdb
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io

def processImage(im_path, transformer):
  data_in = caffe.io.load_image(im_path)
  data_in = caffe.io.resize_image(data_in, (227, 227))
  processed_image = transformer.preprocess('data_in',data_in)
  return processed_image

class ImageProcessor():
  def __init__(self, transformer):
    self.transformer = transformer
  def __call__(self, im_path):
    return processImage(im_path, self.transformer)


def advance_batch(result, N, channels, height, width, idx, num_examples, label, im_paths, image_processor, pool):
    label_r = np.zeros((N))
    im_in_batch = []
    if idx + N >= num_examples:
      label_r[0:num_examples-idx] = label[idx:]
      label_r[num_examples-idx:] = label[0:num_examples-idx]
      im_in_batch[0:num_examples-idx] = im_paths[idx:]
      im_in_batch[num_examples-idx:] = im_paths[0:num_examples-idx]
    else:
      label_r = label[idx:idx+N]
      im_in_batch = im_paths[idx:idx+N]
  
#    data = np.zeros(np.array((N, channels, height, width))[[0,1,2,3]], dtype=np.float32)
#    for ix, im_path in enumerate(im_in_batch):
#      data_in = caffe.io.load_image(im_path)
#      data_in = caffe.io.resize_image(data_in, (227, 227))
#      data[ix] = transformer.preprocess('data_in',data_in)
    
#    pdb.set_trace()
    result['data'] = pool.map(image_processor, im_in_batch)
    
    result['label'] = label_r

#    result['batch'] = generator.get_next_batch(truncate_at_exhaustion=False)
#    image_list = [filename for filename, _ in generator.image_list]
#    result['images'] = pool.map(image_processor, image_list)

class BatchAdvancer():
    def __init__(self, result, label, im_paths, num_examples, N, channels, height, width, pool, idx, transformer):
      self.result = result
      self.label = label
      self.im_paths = im_paths
      self.num_examples = num_examples
      self.N = N
      self.channels = channels
      self.height = height
      self.width = width
      self.pool = pool
      self.idx = idx
      self.transformer = transformer
      self.image_processor = ImageProcessor(self.transformer)
 
    def __call__(self):
      idx_in = self.idx
      self.idx += self.N
      if self.idx >= self.num_examples:
        self.idx = 0
      return advance_batch(self.result, self.N, self.channels, self.height, self.width, idx_in, self.num_examples, self.label, self.im_paths, self.image_processor, self.pool)

class imageRead(caffe.Layer):

  def initialize(self):
    self.train_or_test = 'train'
    self.N = 128
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = ''
    self.image_list = '/home/lisaanne/caffe-forward-backward/examples/pose_Georgia/JHMDB_actions/JHMDB_frames_no_bb_try1_train.txt' 

  def setup(self, bottom, top):
    self.initialize()
    f = open(self.image_list, 'r')
    f_lines = f.readlines()
    f.close()
    self.im_paths = []
    self.label = []
    for line in f_lines:
      self.im_paths.append(line.split(' ')[0])
      self.label.append(int(line.split(' ')[1]))
    self.num_examples = len(self.im_paths)

    #set up data transformer
    shape = (self.N, self.channels, self.height, self.width)
        
    self.transformer = caffe.io.Transformer({'data_in': shape})
    self.transformer.set_raw_scale('data_in', 255)
    image_mean = [103.939, 116.779, 128.68]
    channel_mean = np.zeros((3,227,227))
    for channel_index, mean_val in enumerate(image_mean):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data_in', channel_mean)
    self.transformer.set_channel_swap('data_in', (2, 1, 0))
    self.transformer.set_transpose('data_in', (2, 0, 1))

    self.thread_result = {}
    self.thread = None
    pool_size = 24
    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancer(self.thread_result, self.label, self.im_paths, self.num_examples, self.N, self.channels, self.height, self.width, self.pool, self.idx, self.transformer)
    self.dispatch_worker()
    self.top_names = ['data', 'label']
    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'data':
        shape = (self.N, self.channels, self.height, self.width)
    
    
      elif name == 'label':
        shape = (self.N,)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
   
    if self.thread is not None:
      self.join_worker() 

    for top_index, name in zip(range(len(top)), self.top_names):
      #pdb.set_trace()
      if name == 'data':
        for i in range(self.N):
          top[top_index].data[i, ...] = self.thread_result['data'][i] 
      elif name == 'label':
        top[top_index].data[...] = self.thread_result['label']
 
    self.dispatch_worker()
      

 
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propagate_down, bottom):
    pass

class imageReadTrain(imageRead):

  def initialize(self):
    self.train_or_test = 'train'
    self.N = 128
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = ''
    self.image_list = '/home/lisaanne/caffe-forward-backward/examples/pose_Georgia/JHMDB_actions/JHMDB_frames_no_bb_try1_train.txt' 

class imageReadTest(imageRead):

  def initialize(self):
    self.train_or_test = 'test'
    self.N = 128
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = ''
    self.image_list = '/home/lisaanne/caffe-forward-backward/examples/pose_Georgia/JHMDB_actions/JHMDB_frames_no_bb_try1_test.txt' 

class HDF5Read(caffe.Layer):

    def initialize(self):
      self.train_or_test = 'train'
      self.T = 40
      self.N = 25
      self.F = 30
      self.idx = 0
      self.HDF_File = 'norm32BUG_reshape_train_try2.h5'

    def setup(self, bottom, top):
      self.initialize() 
      #pdb.set_trace() 
      f = h5py.File(self.HDF_File)

      self.clip_markers = np.array(f['clip_markers'])
      self.label = np.array(f['label'])
      self.joints = np.array(f['joints'])
      self.num_examples = self.label.shape[1]
 
      self.top_names = ['joints','labels','clip_markers']
      print 'Outputs:', self.top_names
      if len(top) != len(self.top_names):
        raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                        (len(self.top_names), len(top)))
        
      for top_index, name in enumerate(self.top_names):
        if name == 'joints':
          shape = (self.T,self.N,self.F,1)  
        elif name == 'clip_markers':
          shape = (self.T,self.N)
        else:
          shape = (self.T, self.N) 
        top[top_index].reshape(*shape)
    
    def reshape(self, bottom, top):
      pass
    
    def forward(self, bottom, top):
      

      joints_final = np.zeros((self.T, self.N, self.F,1))
      labels_final = np.zeros((self.T, self.N))
      clip_markers_final = np.zeros((self.T, self.N))
      weight_loss_final = np.zeros((self.T, self.N))

      clip_markers_final = self.clip_markers[:,self.idx:self.idx+self.N]
      joints_final[:,:,:,0] = self.joints[:,self.idx:self.idx+self.N,:]
      label = self.label[:,self.idx:self.idx+self.N] 

      self.idx += self.N
      if self.idx >= self.num_examples:
        self.idx = 0

      for top_index, name in zip(range(len(top)), self.top_names):
        if name == 'joints':
          top[top_index].data[...] = joints_final
        elif name == 'clip_markers':
          top[top_index].data[...] = clip_markers_final
        elif name == 'labels':
          top[top_index].data[...] = labels_final
    
    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def backward(self, top, propagate_down, bottom):
        pass

class HDF5ReadTrain(HDF5Read):
  
    def initialize(self):
      self.train_or_test = 'train'
      self.T = 40
      self.N = 25
      self.F = 30
      self.idx = 0
      self.HDF_File = 'norm32BUG_reshape_train_try2.h5'

class HDF5ReadTest(HDF5Read):
  
    def initialize(self):
      self.train_or_test = 'test'
      self.T = 40
      self.N = 5
      self.F = 30
      self.idx = 0
      self.HDF_File = 'norm32BUG_reshape_test.h5'

##############################################################
class JointsSequenceLayer(caffe.Layer):
  
    def initialize(self):
      self.train_or_test = 'train'
      self.T = 40
      self.N = 25
      self.F = 30
      self.idx = 0
 
    def setup(self, bottom, top):  #will output a train_list and test_list
      self.initialize()
      self.idx = 0
      self.split = 1
      self.train_test_folder = '/mnt/y/lisaanne/JHMDB/splits'


      train_test_files = glob.glob('%s/*%d.txt' %(self.train_test_folder, self.split))

      train_list = [] #will be tuples with (video, label) 
      test_list = [] 

      action_hash = pkl.load(open('/mnt/y/lisaanne/JHMDB/action_hash.p','rb'))
      

      for train_test_file in train_test_files:
        f = open(train_test_file, 'rb')
        lines = f.readlines()
        action = train_test_file.split('/')[6].split('_test')[0]
        label = action_hash[action] 
        for line in lines:
          video = line.split(' ')[0].replace('.avi','')
          is_train = line.split(' ')[1]
          if int(is_train) == 1:
            train_list.append((video, label))
          else:
            test_list.append((video, label))
        
      
      random.shuffle(train_list)
      random.shuffle(test_list)

      if self.train_or_test == 'train':
        self.video_list = train_list
      else:
        self.video_list = test_list 

      self.top_names = ['joints','labels','clip_markers']
      print 'Outputs:', self.top_names
      if len(top) != len(self.top_names):
        raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                        (len(self.top_names), len(top)))
        
      for top_index, name in enumerate(self.top_names):
        if name == 'joints':
          shape = (self.T,self.N,self.F,1)  
        elif name == 'clip_markers':
          shape = (self.T,self.N)
        else:
          shape = (self.T, self.N) 
        top[top_index].reshape(*shape)

    def reshape(self, bottom, top):
      pass

    def forward(self, bottom, top):
      base_gt = '/mnt/y/lisaanne/JHMDB/joint_positions/%s/%s/joint_positions.mat'

      joints_final = np.zeros((self.T, self.N, self.F, 1))
      labels_final = np.zeros((self.T, self.N))
      clip_markers_final = np.zeros((self.T, self.N))
      weight_loss_final = np.zeros((self.T, self.N))
        

      action_hash_rev = pkl.load(open('/mnt/y/lisaanne/JHMDB/action_hash_rev.p','rb'))

      for i in range(0,self.N):
        #import raw joints from video[i]
        video = self.video_list[self.idx][0]
        label = self.video_list[self.idx][1]
        joints_raw = scipy.io.loadmat(base_gt %(action_hash_rev[label], video))

        #do preprocessing for joints

        #norm32
        joints_norm = normalize_joint_x_y(joints_raw['pos_img'])
        joints_processed = np.concatenate((joints_norm[0,:,:],joints_norm[1,:,:])).T

        joints_final[0:joints_processed.shape[0],i,:,0] = joints_processed
        labels_final[:,i] = label
        labels_final[0:self.T/2,i] = -1  #Weight loss
        clip_markers_final[:,i] = 1
        clip_markers_final[0,i] = 0
        
        #increment self.idx
        self.idx += 1
        if self.idx >= len(self.video_list):
          self.idx = 0

      for top_index, name in zip(range(len(top)), self.top_names):
        if name == 'joints':
          top[top_index].data[...] = joints_final
        elif name == 'clip_markers':
          top[top_index].data[...] = clip_markers_final
        elif name == 'labels':
          top[top_index].data[...] = labels_final
     
#      if self.train_or_test == 'train': 
#        pdb.set_trace()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def backward(self, top, propagate_down, bottom):
        pass


class JointsSequenceLayerTrain(JointsSequenceLayer):
  
    def initialize(self):
      self.train_or_test = 'train'
      self.T = 40
      self.N = 25
      self.F = 30
      self.idx = 0

class JointsSequenceLayerTest(JointsSequenceLayer):
  
    def initialize(self):
      self.train_or_test = 'test'
      self.T = 40
      self.N = 5
      self.F = 30
      self.idx = 0

