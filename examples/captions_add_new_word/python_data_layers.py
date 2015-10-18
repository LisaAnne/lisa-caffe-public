#!/usr/bin/env python

#Data layer for video.  Change flow_frames and RGB_frames to be the path to the flow and RGB frames.

import sys
sys.path.append('../../python')
import caffe
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
import copy
import json
import hickle as hkl
from init import *
import time

vocabulary_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'
num_image_features = 471

def split_sent(sent):
  re.sub('[^A-Za-z0-9\s]+','',sent)
  return sent

def processCaptions(captions, vocabulary, stream):
  caption_split = split_sent(caption)
  caption_numbers = [vocab_lines.index(w) for w in caption_split]
  num_words = len(caption_numbers)

  input_sentence = [-1]*stream
  target_sentence = [-1]*stream

  input_sentence[0] = 0
  input_sentence[1:num_words+1] = caption_numbers
  target_sentence[:num_words] = caption_numbers
  target_sentence[num_words] = 0

  return input_sentence, target_sentence 

class CaptionProcessor(object):
  def __init__(self, stream):
    vocab_lines = open(vocabulary_file).readlines()
    vocab_lines = [v.strip() for v in vocab_lines]
    vocab_lines = ['EOS'] + vocab_lines
    self.vocab_lines = vocab_lines
    self.stream = stream

  def __call__(self, captions):
    return processCaptions(captions, self.vocabulary, self.stream)

def processImage(im_path, transformer):
  data_in = caffe.io.load_image(im_path)
  data_in = caffe.io.resize_image(data_in, (256, 256))
  crop_x = random.randint(0, 256-self.dim)
  crop_y = random.randint(0, 256-self.dim)
  data_in = data_in[crop_x:crop_x+self.dim, crop_y:crop_y+self.dim,:]
  processed_image = transformer.preprocess('data_in',data_in)
  return processed_image

class ImageProcessor(object):
  def __init__(self, transformer, dim):
    self.transformer = transformer
    self.dim = dim
  def __call__(self, im_path):
    return processImage(im_path, self.transformer, self.dim)

class sequenceGenerator(object):
  def __init__(self, stream, buffer_size):
    self.stream = stream
    self.buffer_size = buffer_size

  def __call__(self):

    #take 100 random images
    sample_images = random.sample(range(len(self.images_json)), self.batch_size)
    labels = np.zeros(self.batch_size, num_image_features)
    for i in range(self.batch_size):
      im_id = int(im.split('/')[-1].split('_')[-1].split('.jpg')[0])
      labels[i,:] = self.image_labels[im_id]
    images = [i['image_id'] for i in sample_images['images']]  

    sample_captions = random.sample(range(len(self.captions_json)), self.batch_size)
    captions = [a['caption'] for a in sample_captins['annotations']]

    sample_paired = random.sample(range(len(self.paired_json)), self.batch_size)
    paired_images = [i['image_id'] for i in sample_images['images']]  
    paired_captions = [a['caption'] for a in sample_captins['annotations']]
    
    return images, labels, captions, paired_images, paired_captions 
  
def advance_batch(result, sequence_generator, image_processor, sentence_processer, pool):
 
    #use image processor to process two sets of images
    images, labels, captions, paired_images, paired_captions = sequence_generator()
    
    result['images'] = pool.map(image_processor, images)
    result['image_label'] = labels
    result['paired-images'] = pool.map(image_processor, paired_images)
    result['input_sentence'], result['target_sentence'] = pool.map(sentence_processor, captions)
    result['paired-input_sentence'], result['paired-target_sentence'] = pool.map(sentence_processor, paired_captions)

    cm = np.ones(input_sentence.shape)
    cm[0::stream,:] = 0
    result['cont_sentence'] = cm
    result['paired-cont_sentence'] = cm

class BatchAdvancer():
    def __init__(self, result, sequence_generator, sentence_processor, image_processor, pool):
      self.result = result
      self.sequence_generator = sequence_generator
      self.image_processor = image_processor
      self.sentence_processor = sentence_processor
      self.pool = pool
 
    def __call__(self):
      return advance_batch(self.result, self.sequence_generator, self.image_processor, self.sentence_processor, self.pool)

class BatchAdvancerImage():
    def __init__(self, result, image_processor, images_with_labels, batch_size, pool):
      self.result = result
      self.image_processor = image_processor
      self.pool = pool
      self.images_with_labels = images_with_labels
      self.image_order = self.images_with_labels.keys()
      self.batch_size = batch_size
      self.iteration = 0*100

    def __call__(self):
      
      if self.iteration + self.batch_size < len(self.image_order):    
        next_batch_images = self.image_order[self.iteration:self.iteration+self.batch_size]
        next_batch_labels = [self.images_with_labels[im] for im in self.image_order[self.iteration:self.iteration+self.batch_size]] 
      else:
        next_batch_images = [None]*self.batch_size
        next_batch_labels = [None]*self.batch_size
        end_current_iter = len(self.image_order[self.iteration:]) 
        next_batch_images[:end_current_iter] = self.image_order[self.iteration:]
        next_batch_labels[:end_current_iter] = [self.images_with_labels[im] for im in self.image_order[self.iteration:]] 
        random.shuffle(self.image_order)
        next_batch_images[end_current_iter:] = self.image_order[:self.batch_size-end_current_iter]
        next_batch_labels[end_current_iter:] = [self.images_with_labels[im] for im in self.image_order[:self.batch_size-end_current_iter]]
        self.iteration = self.batch_size-end_current_iter 
 
      if self.pool:
        self.result['images'] = self.pool.map(self.image_processor, next_batch_images)
      else:
        self.result['images'] = []
        for nbi in next_batch_images:
          #pdb.set_trace()
          self.result['images'].append(self.image_processor(nbi))
      self.result['labels'] = next_batch_labels 
      self.iteration += self.batch_size

class BatchAdvancerFeature():
    def __init__(self, result, images_with_labels, batch_size, pool, shuffle=True, image_order=None):
      self.result = result
      self.pool = pool
      self.images_with_labels = images_with_labels
      if image_order is None:
        self.image_order = self.images_with_labels.keys()
      else:
        self.image_order = image_order
      self.batch_size = batch_size
      self.iteration = 0*100
      self.shuffle = shuffle

    def __call__(self):
      
      if self.iteration + self.batch_size < len(self.image_order):    
        next_batch_features = [self.images_with_labels[im]['features'] for im in self.image_order[self.iteration:self.iteration+self.batch_size]] 
        next_batch_labels = [self.images_with_labels[im]['labels'] for im in self.image_order[self.iteration:self.iteration+self.batch_size]] 
        self.iteration += self.batch_size
      else:
        next_batch_features = [None]*self.batch_size
        next_batch_labels = [None]*self.batch_size
        end_current_iter = len(self.image_order[self.iteration:]) 
        next_batch_features[:end_current_iter] = [self.images_with_labels[im]['features'] for im in self.image_order[self.iteration:]]
        next_batch_labels[:end_current_iter] = [self.images_with_labels[im]['labels'] for im in self.image_order[self.iteration:]]
        next_batch_features[end_current_iter:] = [self.images_with_labels[im]['features'] for im in self.image_order[:self.batch_size-end_current_iter]]
        next_batch_labels[end_current_iter:] = [self.images_with_labels[im]['labels'] for im in self.image_order[:self.batch_size-end_current_iter]]
        #pdb.set_trace()
        #if self.shuffle:
        #  random.shuffle(self.image_order)
        self.iteration = self.batch_size-end_current_iter 
 
      self.result['features'] = next_batch_features
      self.result['labels'] = next_batch_labels 

class featureDataLayer(caffe.Layer):

  def setup(self, bottom, top):
    random.seed(10)
    
    #create dict with all features (will be an issue if there are too many features or if using lower level feature like conv5)
    
    dataset_path_hash = {'coco': coco_root, 'imagenet': imagenet_root} 

    param_str = eval(self.param_str)
    self.batch_size = param_str['batch_size']
    self.extracted_features = param_str['extracted_features']
    self.image_list = param_str['image_list']
    self.feature_size = param_str['feature_size'] 

    #read in all extracted features and sort into single dict
    images_with_labels = {}
    extracted_features = []
    #read h5py
    extracted_features = h5py.File(self.extracted_features,'r')
    
    t = time.time()
    for ix, im in enumerate(extracted_features['ims']):
      im_key = im.split('_')[-1].split('.jpg')[0]
      images_with_labels[im_key] = {} 
      images_with_labels[im_key]['features'] = extracted_features['features'][ix]
      images_with_labels[im_key]['labels'] = int(im_key)
    print "Setting up images dict: ", time.time()-t
    
    del extracted_features
 
    images = open(self.image_list, 'rb').readlines()
    self.images = [im.strip() for im in images]
    self.images = [im.split('_')[-1].split('.jpg')[0] for im in self.images]

    #set up thread and batch advancer
    self.thread_result = {}
    self.thread = None
    pool_size = 4

    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancerFeature(self.thread_result, images_with_labels, self.batch_size, self.pool, shuffle=False, image_order=self.images)
    self.dispatch_worker()

    self.top_names = ['features', 'labels']

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'features':
        shape = (self.batch_size, self.feature_size)
      else:
        shape = (self.batch_size, 1)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    for top_index, name in zip(range(len(top)), self.top_names):
      for i in range(self.batch_size):
        top[top_index].data[i, ...] = self.thread_result[name][i] 

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

class captionClassifierFeatureData(caffe.Layer):

  def initialize(self):
    self.batch_size = 100
    self.feature_size = 4096
    self.json_images = 'utils_trainAttributes/imageJson_test.json' 
    self.images = 'utils_trainAttributes/imageVal_zebraImagenet.txt' #txt file organized as: dataset image_path
    self.lexical_classes = 'utils_trainAttributes/lexicalList_parseCoco_JJ100_NN300_VB100.txt' #txt file
    self.single_bit_classes = ['zebra'] #txt file
    self.extracted_features = {'coco': '/y/lisaanne/vgg_features/h5Files/coco2014_cocoid.train.txt0.h5', 'imagenet': '/y/lisaanne/vgg_features/h5Files/vgg_feats_imagenet_zebra.h5'}

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    
    #create dict with all features (will be an issue if there are too many features or if using lower level feature like conv5)
    
    dataset_path_hash = {'coco': coco_root, 'imagenet': imagenet_root} 
  
    self.params = eval(self.param_str)
    if 'batch_size' in self.params:
      self.batch_size = self.params['batch_size']

    #read in all extracted features and sort into single dict
    images_with_labels = {}
    extracted_features = []
    #read h5py
    for key in self.extracted_features:
      extracted_features.append([key, h5py.File(self.extracted_features[key],'r')])
    
    t = time.time()
    for dset, ef in extracted_features:
      for ix, im in enumerate(ef['ims']):
    
        images_with_labels[dataset_path_hash[dset] + im] = {}
        images_with_labels[dataset_path_hash[dset] + im]['features'] = ef['features'][ix]
        images_with_labels[dataset_path_hash[dset] + im]['labels'] = None
    print "Setting up images dict: ", time.time()-t
    
    del extracted_features
 
    images = open(self.images, 'rb').readlines()
    images = [(im.split(' ')[0], im.split(' ')[1].strip()) for im in images]

    #read json
    def read_json(t_file):
      j_file = open(t_file).read()
      return json.loads(j_file)

    lexical_classes = open(self.lexical_classes, 'rb').readlines()
    lexical_classes = [i.strip() for i in lexical_classes]
    lexical_classes_dict = {}
    for ix, lexical_class in enumerate(lexical_classes):
      lexical_classes_dict[lexical_class] = ix
    single_bit_classes_idx = [lexical_classes_dict[single_bit_class] for single_bit_class in self.single_bit_classes]
    single_bit_classes_inverse_idx = [lexical_classes_dict[single_bit_class] for single_bit_class in lexical_classes if single_bit_class not in self.single_bit_classes] 

    def filter_labels(label_list):
      final_labels = np.ones(len(lexical_classes),)*-1
      positive_idx = [lexical_classes_dict[label] for label in label_list['positive_label']]
      negative_idx = [lexical_classes_dict[label] for label in label_list['negative_label']]
      final_labels[positive_idx] = 1
      final_labels[negative_idx] = 0
      if (single_bit_classes_idx > 0) and (np.sum(final_labels[single_bit_classes_idx]) > 0):
        final_labels[single_bit_classes_inverse_idx] = -1 
      return final_labels

    t = time.time()
    json_images = read_json(self.json_images)
    print 'Reading json image dicts takes: ', time.time() - t

    #filter image labels and set up 
    t = time.time()
    for dset, path in images:
      labels = filter_labels(json_images['images'][dset][path])
      if 'val' in path:
        path = path.replace('val', 'trainval')
      else:
        path = path.replace('train', 'trainval')
      images_with_labels[dataset_path_hash[dset] + path]['labels'] = labels
    keys = images_with_labels.keys()
    for key in keys:
      if images_with_labels[key]['labels'] is None:
        images_with_labels.pop(key, None)

    print 'Filtering labels takes: ', time.time() - t

    #set up thread and batch advancer
    self.thread_result = {}
    self.thread = None
    pool_size = 4

    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancerFeature(self.thread_result, images_with_labels, self.batch_size, self.pool)
    self.dispatch_worker()

    self.top_names = ['features', 'labels']

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'features':
        shape = (self.batch_size, self.feature_size)
      else:
        shape = (self.batch_size, len(lexical_classes))
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    for top_index, name in zip(range(len(top)), self.top_names):
      for i in range(self.batch_size):
        top[top_index].data[i, ...] = self.thread_result[name][i] 

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

class captionClassifierImageData(caffe.Layer):

  #For this data layer must have the following things defined:
  #  A json dict of images (json_images) which has image paths as well as image labels
  #  A list of images which will be used (do not need to use all images in json dict)
  #  A list of classes to be classified.
  #  Words which will only get a single bit label.

  def setup(self, bottom, top):
    random.seed(10)

    self.channels = 3
    self.lexical_classes = 'utils_trainAttributes/lexicalList_parseCoco_JJ100_NN300_VB100.txt' #txt file

    dataset_path_hash = {'coco': coco_root, 'imagenet': imagenet_root} 

    #read json
    def read_json(t_file):
      j_file = open(t_file).read()
      return json.loads(j_file)

    self.params = eval(self.param_str)
    assert 'batch_size' in self.params.keys(), 'Params must include batch size.'
    assert 'single_bit_classes' in self.params.keys(), 'Params must include single bit classes.'
    assert 'images' in self.params.keys(), 'Params must include list of images.'
    assert 'crop_dim' in self.params.keys(), 'Params must include crop_dim.'
    self.batch_size = self.params['batch_size']
    self.single_bit_classes = self.params['single_bit_classes']
    self.images = self.params['images']
    self.height = self.params['crop_dim']
    self.width = self.params['crop_dim']

    if 'batch_size' in self.params:
      self.batch_size = self.params['batch_size']
    lexical_classes = open(self.lexical_classes, 'rb').readlines()
    lexical_classes = [i.strip() for i in lexical_classes]
    lexical_classes_dict = {}
    for ix, lexical_class in enumerate(lexical_classes):
      lexical_classes_dict[lexical_class] = ix
    single_bit_classes_idx = [lexical_classes_dict[single_bit_class] for single_bit_class in self.single_bit_classes]
    single_bit_classes_inverse_idx = [lexical_classes_dict[single_bit_class] for single_bit_class in lexical_classes if single_bit_class not in self.single_bit_classes] 

    def filter_labels(label_list):
      final_labels = np.ones(len(lexical_classes),)*-1
      positive_idx = [lexical_classes_dict[label] for label in label_list['positive_label']]
      negative_idx = [lexical_classes_dict[label] for label in label_list['negative_label']]
      final_labels[positive_idx] = 1
      final_labels[negative_idx] = 0
      if (single_bit_classes_idx > 0) and (np.sum(final_labels[single_bit_classes_idx]) > 0):
        final_labels[single_bit_classes_inverse_idx] = -1
         
      return final_labels

    t = time.time()
    json_images = read_json(self.json_images)
    print 'Reading json image dicts takes: ', time.time() - t

    #filter image labels and set up
    images = open(self.images, 'rb').readlines()
    images = [(im.split(' ')[0], im.split(' ')[1].strip()) for im in images]
    images_with_labels = {}
    t = time.time()
    for dset, path in images:
      pdb.set_trace()
      labels = filter_labels(json_images['images'][dset][path])
      images_with_labels[dataset_path_hash[dset] + path] = labels
    print 'Filtering labels takes: ', time.time() - t

    #set up data transformer
    shape = (self.batch_size, self.channels, self.height, self.width)
        
    self.transformer = caffe.io.Transformer({'data_in': shape})
    self.transformer.set_raw_scale('data_in', 255)
    image_mean = [103.939, 116.779, 128.68]
    channel_mean = np.zeros((3,self.height,self.width))
    for channel_index, mean_val in enumerate(image_mean):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data_in', channel_mean)
    self.transformer.set_channel_swap('data_in', (2, 1, 0))
    self.transformer.set_transpose('data_in', (2, 0, 1))

    #set up thread and batch advancer
    self.thread_result = {}
    self.thread = None
    pool_size = 4

    self.image_processor = ImageProcessor(self.transformer, self.height)

    if pool_size > 0:
      self.pool = Pool(processes=pool_size)
    else:
      self.pool = None
    self.batch_advancer = BatchAdvancerImage(self.thread_result, self.image_processor, images_with_labels, self.batch_size, self.pool)
    self.dispatch_worker()

    self.top_names = ['images', 'labels']

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'images':
        shape = (self.batch_size, self.channels, self.height, self.width)
      else:
        shape = (self.batch_size, len(lexical_classes))
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    for top_index, name in zip(range(len(top)), self.top_names):
      for i in range(self.batch_size):
        top[top_index].data[i, ...] = self.thread_result[name][i] 

    pdb.set_trace()
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




#############################################

class dataRead(caffe.Layer):

  def initialize(self):
    self.stream = 20
    self.buffer_size = 100
    self.batch_size = 100 
    self.channels = 3
    self.path_to_images = '' 
    self.captions_json_file = ''
    self.images_h5_file = ''
    self.paired_json_file = ''
    self.image_label_json_file = ''
    self.image_label_shape = (self.batch_size, num_image_features)

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()


    #read json
    def read_json(t_file):
      j_file = open(t_file).read()
      return json.loads(j_file)

    self.captions_json = read_json(self.captions_json_file)
    self.images_json = h5py.File(self.images_h5_file, 'r')
    self.paired_json = read_json(self.paired_json_file)

    #set up data transformer
    shape = (self.batch_size, self.channels, self.height, self.width)
        
    self.transformer = caffe.io.Transformer({'data_in': shape})
    self.transformer.set_raw_scale('data_in', 255)
    image_mean = [103.939, 116.779, 128.68]
    channel_mean = np.zeros((3,self.height,self.width))
    for channel_index, mean_val in enumerate(image_mean):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data_in', channel_mean)
    self.transformer.set_channel_swap('data_in', (2, 1, 0))
    self.transformer.set_transpose('data_in', (2, 0, 1))

    self.thread_result = {}
    self.thread = None
    pool_size = 12

    self.image_processor = ImageProcessorCrop(self.transformer, self.height)
    self.sentence_processor = CaptionProcessor(self.stream) 
    self.sequence_generator = sequenceGenerator(self.stream, self.buffer_size)

    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.sentence_processor, self.image_processor, self.pool)
    self.dispatch_worker()

    self.top_names = ['images', 'image_label', 'input_sentence', 'cont_sentnece', 'target_sentence', 'paired-images','paired-input_sentence', 'paired-cont_sentence', 'paired-target_sentence']

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'images':
        shape = (self.batch_size, self.channels, self.height, self.width)
      elif name in ['captions', 'cont_sentence', 'target_sentence', 'paired-captions', 'paired-cont_sentence', 'paired-target_sentence']:
        shape = (self.stream, self.buffer_size)
      elif name == 'image_label':
        shape = self.image_label_shape
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    for top_index, name in zip(range(len(top)), self.top_names):
      if name in ['images', 'paired-images', 'input_sentence', 'target_sentence', 'paired-input_sentence', 'paired-target_sentence']:
        for i in range(self.N):
          top[top_index].data[i, ...] = new_result_data[i] 
      else:
        top[top_index].data[...] = self.thread_result[name]

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

class dataRead_general(dataRead):

  def initialize(self):
    captions_tag = 'train'
    image_tag = 'zebra'
    paired_tag = 'no_caption_zebra_train'
    
    self.initialize_general(captions_tag, image_tag, paired_tag) 

  def initialize_general(self, captions_tag, image_tag, paired_tag):

    self.stream = 20
    self.buffer_size = 100
    self.batch_size = 100 
    self.channels = 3
    self.path_to_images = '../../data/coco/coco/images/trainval2014' 
    self.captions_json_file = '../../data/coco/coco/annotations/captions_%s2014.json' %captions_tag
    self.images_h5_file = '/x/lisaanne/coco_attribute/utils_trainAttributes/attributes_rm%s_JJ100_NN300_VB100_train.h5' %image_tag
    self.paired_json_file = '../../data/coco/coco/annotations/captions_%s2014.json' %paired_tag
    self.image_label_shape = (self.batch_size, num_image_features)

class dataRead_zebra_train(dataRead_general):
  def initialize(self):
    #captions_tag points to json files which has sentences language model must reconstruct
    captions_tag = 'train'
    #image_tag points to json files which has images and image labels
    image_tag = 'zebra'
    #paired_tag points to json file which has paired data
    paired_tag = 'no_caption_zebra_train'
        
    
    self.initialize_general(captions_tag, image_tag, paired_tag) 

#class dataRead_zebra_test(dataRead_general):
#  def initialize(self):
#    captions_tag = 'val_val2014'
#    image_tag = 'val_val2014'
#    paired_tag = 'no_caption_zebra_train'
#    
#    self.initialize_general(captions_tag, image_tag, paired_tag) 
# can just use normal evaluation
