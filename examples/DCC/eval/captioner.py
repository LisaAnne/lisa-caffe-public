#!/usr/bin/env python

from collections import OrderedDict
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import pickle as pkl
import copy
import time
import pdb
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

home_dir = '/home/lisaanne/caffe-LSTM/'
sys.path.append(home_dir + '/python/')
import caffe

from utils.config import *
from utils.python_utils import *
from utils.caffe_utils import *

def coco_fill(split, i):
  return coco_template %(split, split, int(i))

def imagenet_fill(split, i):
  return i 

template = {'coco': coco_fill, 'imagenet': imagenet_fill}

class Captioner():
  def __init__(self, weights_path, image_net_proto, lstm_net_proto, 
               vocab_path, device_id=1, precomputed_feats=None, 
	       prev_word_restriction=True, image_feature='probs', language_feature='probs'):
    if device_id >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(device_id)
    else:
      caffe.set_mode_cpu()

    self.image_feature = image_feature
    self.language_feature = language_feature

    # Setup image processing net.
    phase = caffe.TEST
    self.prev_word_bool = prev_word_restriction
    #Assume the image weights are the same for all models
    self.image_net = None
    self.weights_name = weights_path.split('/')[-1].strip('.caffemodel')
    if image_net_proto:
      self.image_net = caffe.Net(image_net_proto, weights_path, phase)
      image_data_shape = self.image_net.blobs['data'].data.shape
      if image_data_shape[1] == 3:
        self.transformer = build_transformer(image_data_shape[-1]) 
      else:
        print "Warning: did not set transformer; assume that image features do not need to be transformed.\n"
    self.lstm_net = caffe.Net(lstm_net_proto, weights_path, phase)
    with open(vocab_path, 'r') as vocab_file:
      self.vocab = [word.strip() for word in vocab_file.readlines()]
    net_vocab_size = self.lstm_net.blobs[self.language_feature].data.shape[-1]
    self.vocab = ['<EOS>'] + self.vocab
    if len(self.vocab) != net_vocab_size:
      raise Exception('Invalid vocab file: contains %d words; '
          'net expects vocab with %d words' % (len(self.vocab), net_vocab_size))

    self.descriptors = None
    self.h5_file = precomputed_feats
    if self.h5_file:
      #load image from h5 file and make into a dict
      extracted_features = {}
      file_type = None
      if self.h5_file.split('.')[-1] == 'h5':
        file_type = 'h5'
        f = h5py.File(self.h5_file, 'r')
        for feature, im in zip(f['features'], f['ims']):
          extracted_features[im] = feature
      elif self.h5_file.split('.')[-1] == 'p':
        file_type = 'pkl' 
        extracted_features = pkl.load(open(self.h5_file, 'r'))
      self.extracted_features = extracted_features 

    else:
      self.preprocess_image = lambda x: image_processor(self.transformer, x, False)

  def set_image_batch_size(self, batch_size):
    self.image_net.blobs['data'].reshape(batch_size, *self.image_net.blobs['data'].data.shape[1:])

  def set_caption_batch_size(self, batch_size, unroll=False):
    dim = len(self.lstm_net.blobs['image_features'].data.shape)
    self.lstm_net.blobs['cont_sentence'].reshape(1, batch_size)
    self.lstm_net.blobs['input_sentence'].reshape(1, batch_size)
  
    if dim == 2:
      self.lstm_net.blobs['image_features'].reshape(batch_size,
          *self.lstm_net.blobs['image_features'].data.shape[1:])
    elif dim == 3:
      self.lstm_net.blobs['image_features'].reshape(1, batch_size,
          *self.lstm_net.blobs['image_features'].data.shape[2:])
  
    if unroll:
      self.lstm_net.blobs['lstm1_h0'].reshape(1, batch_size, self.lstm_net.blobs['lstm1_h0'].shape[2]) 
      self.lstm_net.blobs['lstm1_c0'].reshape(1, batch_size, self.lstm_net.blobs['lstm1_c0'].shape[2]) 
 
    self.lstm_net.reshape()
 
  def compute_descriptors(self, image_root, image_list, batch_size):
    output_name = self.image_feature

    descriptors_shape = (len(image_list), ) + \
        (self.image_net.blobs[output_name].data.shape[-1], )
    descriptors = np.zeros(descriptors_shape)
    for batch_start_index in range(0, len(image_list), batch_size):
      batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
      current_batch_size = min(batch_size, len(image_list) - batch_start_index)
      batch = np.zeros( (current_batch_size,) + (self.image_net.blobs['data'].data.shape[1:]))
      for batch_index, image_path in enumerate(batch_list):
        if self.h5_file:
          batch[batch_index:(batch_index+1),:] = self.extracted_features[image_path] 
        else:
          batch[batch_index:(batch_index + 1),...] = self.preprocess_image(image_root + image_path)
      print 'Computing descriptors for images %d-%d of %d' % \
          (batch_start_index, batch_start_index + current_batch_size - 1,
           len(image_list))
      self.set_image_batch_size(current_batch_size)
      self.image_net.blobs['data'].data[...] = batch
      self.image_net.forward()
      descriptors[batch_start_index:(batch_start_index + current_batch_size)] = \
          self.image_net.blobs[output_name].data[:current_batch_size]
    return descriptors

  def compute_all_descriptors(self, image_root=None, image_ids=None, batch_size=100, file_load=False, split='val', dset='coco'):
    descriptor_filename = '%s/descriptors_%s.npz' % (image_features, self.weights_name)
    if os.path.exists(descriptor_filename) & file_load:
      self.descriptors = np.load(descriptor_filename)['descriptors']
      self.image_ids = np.load(descriptor_filename)['image_ids']
      self.image_list = [template[dset](split, i) for i in image_ids]
    else:
      assert image_ids
      self.iamge_ids = image_ids
      self.image_list = [template[dset](split, i) for i in image_ids] 
      self.descriptors = self.compute_descriptors(image_root, self.image_list, batch_size)
      np.savez_compressed(descriptor_filename, descriptors=self.descriptors, image_ids=image_ids)
  
  def debug_generation(self, descriptor, 
                      temp=1, max_length=50, min_length=2,
                      blob_im='predict-im', blob_lm='predict-lm', 
                      blob_multi='predict-multimodal'):
    net = self.lstm_net
    prob_output_name = self.language_feature

    descriptor = np.array(descriptor)
    batch_size = descriptor.shape[0]
    self.set_caption_batch_size(batch_size, False)
    prev_word_bool = self.prev_word_bool

    cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
    word_input = np.zeros_like(net.blobs['input_sentence'].data)
    image_features = np.zeros_like(net.blobs['image_features'].data)

    image_features[:] = descriptor
    outputs = []
    output_captions = [[] for b in range(batch_size)]
    output_probs = [[] for b in range(batch_size)]
    caption_index = 0
    num_done = 0


    predict_lm = np.zeros((10,1,len(self.vocab)))
    predict_multi = np.zeros((10,1,len(self.vocab)))

    while caption_index < 10:
      if caption_index == 0:
        cont_input[:] = 0
      elif caption_index == 1:
        cont_input[:] = 1
      if caption_index == 0:
        word_input[:] = 0
      else:
        for index in range(batch_size):
          word_input[0, index] = \
              output_captions[index][caption_index - 1] if \
              caption_index <= len(output_captions[index]) else 0
      net.blobs['image_features'].data[...] = image_features
      net.blobs['cont_sentence'].data[...] = cont_input
      net.blobs['input_sentence'].data[...] = word_input
      net.forward()
      predict_lm[caption_index,:,:] = net.blobs[blob_lm].data
      predict_multi[caption_index,:,:] = net.blobs[blob_multi].data
      net_output_probs = net.blobs[prob_output_name].data[0]
 
      #generation tracking stuff
      if temp == 1.0 or temp == float('inf'):
        no_EOS = False if (caption_index > min_length) else True
        if prev_word_bool & caption_index > 0:
          prev_words = [output_caption[-1] for output_caption in output_captions]
        else: prev_words = [False]*len(output_captions)
        samples = [
            random_choice_from_probs(dist, temp=temp, already_softmaxed=True, no_EOS=no_EOS, prev_word=prev_word)
            for dist, prev_word in zip(net_output_probs, prev_words)
        ]
      else:
        samples = [
            random_choice_from_probs(preds, temp=temp, already_softmaxed=False)
            for preds in net_output_probs
        ]
      for index, next_word_sample in enumerate(samples):
        if not output_captions[index] or output_captions[index][-1] != 0:
          output_captions[index].append(next_word_sample)
          output_probs[index].append(net_output_probs[index, next_word_sample])
          if next_word_sample == 0: num_done += 1
      sys.stdout.write('\r%d/%d done after word %d' %
          (num_done, batch_size, caption_index))
      sys.stdout.flush()
      caption_index += 1
    sys.stdout.write('\n')

    return net.blobs[blob_im].data.squeeze(), predict_lm.squeeze(), predict_multi.squeeze(), output_captions[0]

  def sample_captions(self, descriptor, 
                      temp=1, max_length=50, min_length=2):
    net = self.lstm_net
    prob_output_name = self.language_feature
    unroll = False
    if 'lstm1_h0' in net.blobs.keys(): unroll=True

    descriptor = np.array(descriptor)
    batch_size = descriptor.shape[0]
    self.set_caption_batch_size(batch_size, unroll)
    prev_word_bool = self.prev_word_bool

    cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
    word_input = np.zeros_like(net.blobs['input_sentence'].data)
    image_features = np.zeros_like(net.blobs['image_features'].data)
    if unroll:
      hidden_unit = np.zeros_like(net.blobs['lstm1_h0'].data)
      cell_unit = np.zeros_like(net.blobs['lstm1_c0'].data)
      hidden_unit[:] = 0
      cell_unit[:] = 0

    image_features[:] = descriptor
    outputs = []
    output_captions = [[] for b in range(batch_size)]
    output_probs = [[] for b in range(batch_size)]
    caption_index = 0
    num_done = 0

    while num_done < batch_size and caption_index < max_length:
      if caption_index == 0:
        cont_input[:] = 0
      elif caption_index == 1:
        cont_input[:] = 1
      if caption_index == 0:
        word_input[:] = 0
      else:
        for index in range(batch_size):
          word_input[0, index] = \
              output_captions[index][caption_index - 1] if \
              caption_index <= len(output_captions[index]) else 0
      net.blobs['image_features'].data[...] = image_features
      net.blobs['cont_sentence'].data[...] = cont_input
      net.blobs['input_sentence'].data[...] = word_input
      if unroll:
        net.blobs['lstm1_h0'].data[...] = hidden_unit 
        net.blobs['lstm1_c0'].data[...] = cell_unit
      net.forward()
      #pdb.set_trace()
      net_output_probs = net.blobs[prob_output_name].data[0]
      if unroll:
        hidden_unit = net.blobs['lstm1_h1'].data[...]
        cell_unit = net.blobs['lstm1_c1'].data[...]
 
      #generation tracking stuff
      if temp == 1.0 or temp == float('inf'):
        no_EOS = False if (caption_index > min_length) else True
        if prev_word_bool & caption_index > 0:
          prev_words = [output_caption[-1] for output_caption in output_captions]
        else: prev_words = [False]*len(output_captions)
        samples = [
            random_choice_from_probs(dist, temp=temp, already_softmaxed=True, no_EOS=no_EOS, prev_word=prev_word)
            for dist, prev_word in zip(net_output_probs, prev_words)
        ]
      else:
        samples = [
            random_choice_from_probs(preds, temp=temp, already_softmaxed=False)
            for preds in net_output_probs
        ]
      for index, next_word_sample in enumerate(samples):
        if not output_captions[index] or output_captions[index][-1] != 0:
          output_captions[index].append(next_word_sample)
          output_probs[index].append(net_output_probs[index, next_word_sample])
          if next_word_sample == 0: num_done += 1
      sys.stdout.write('\r%d/%d done after word %d' %
          (num_done, batch_size, caption_index))
      sys.stdout.flush()
      caption_index += 1
    sys.stdout.write('\n')

    return output_captions, output_probs

  def sample_all_captions(self, temp=1, max_length=50, min_length=2,
                          strategy={'type': 'beam', 'beam_size': 1}, max_batch_size=1000):
    num_images = len(self.image_list)

    #need logic if we have any other methods other than generation
    do_batches = True
    batch_size = max_batch_size
    
    all_captions = [None] * num_images
    all_probs = [None] * num_images   
 
    for i in range(0, num_images, batch_size):
      sys.stdout.write("\rGenerating captions for image %d/%d" %(i, num_images))
      sys.stdout.flush()
      end_batch = min(i+batch_size, num_images)
      descriptor = self.descriptors[i:end_batch, ...]
      all_captions[i:end_batch], all_probs[i:end_batch] = self.sample_captions(descriptor, temp=temp, max_length=max_length,
                                                                               min_length=min_length)
    return all_captions, all_probs

  def generate_sentences(self, image_root, image_ids, temp=1, strategy={'type': 'beam', 'beam_size': 1}, tag='val_val_beam1', split='val', dset='coco'):
    self.image_ids = image_ids
    self.compute_all_descriptors(image_root, image_ids, dset=dset, file_load=False)
    all_captions, all_probs = self.sample_all_captions(temp=temp, strategy=strategy)
    return self.save_captions(all_captions,tag) 

  def sentence(self, vocab_indices):
    if not vocab_indices:
      print "Had issue with image!"
      return ' '
    sentence = ' '.join([unicode(self.vocab[i], 'utf-8') for i in vocab_indices])
    if not sentence: return sentence
    sentence = sentence[0].upper() + sentence[1:]
    # If sentence ends with ' <EOS>', remove and replace with '.'
    # Otherwise (doesn't end with '<EOS>' -- maybe was the max length?):
    # append '...'
    suffix = ' ' + self.vocab[0]
    if sentence.endswith(suffix):
      sentence = sentence[:-len(suffix)] + '.'
    else:
      sentence += '...'
    return sentence


  def save_captions(self, captions, tag):
    save_name = '%s/%s_%s.json' %(generated_sentences, self.weights_name, tag)
    save_dict = []
    for image_id, caption in zip(self.image_ids, captions):
      try:
        save_dict.append({'image_id': int(image_id), 'caption': self.sentence(caption)})
      except:
        save_dict.append({'image_id': image_id, 'caption': self.sentence(caption)})
    save_json(save_dict, save_name) 
    print "Saved captions to %s." %save_name
    return save_name

def random_choice_from_probs(softmax_inputs, temp='inf', already_softmaxed=False, no_EOS=False, prev_word = None):
  # temperature of infinity == take the max
  # if no_EOS True, then the next word will not be the end of the sentence
  if temp == float('inf'):
    if prev_word:
      softmax_inputs[prev_word] = 0
    if no_EOS:
        return np.argmax(softmax_inputs[1:]) + 1
    else:
      return np.argmax(softmax_inputs)
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?
  
