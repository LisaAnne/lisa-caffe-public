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

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

home_dir = '/home/lisaanne/caffe-LSTM/'
sys.path.append(home_dir + '/python/')
import caffe
feature_path ='/y/lisaanne/image_captioning/coco_features/' 
class Captioner():
  def __init__(self, weights_paths, image_net_proto, lstm_net_proto,
               vocab_path, device_id=0, precomputed_feats=None, prev_word_restriction=False):
    if device_id >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(device_id)
    else:
      caffe.set_mode_cpu()
    # Setup image processing net.
    phase = caffe.TEST
    self.prev_word_bool = prev_word_restriction
    #Assume the image weights are the same for all models
    self.image_net = caffe.Net(image_net_proto, weights_paths[0], phase)
    image_data_shape = self.image_net.blobs['data'].data.shape
    self.transformer = caffe.io.Transformer({'data': image_data_shape})
    channel_mean = np.zeros(image_data_shape[1:])
    channel_mean_values = [104, 117, 123]
    #assert channel_mean.shape[0] == len(channel_mean_values)
    if channel_mean.shape[0] == len(channel_mean_values):
      for channel_index, mean_val in enumerate(channel_mean_values):
        channel_mean[channel_index, ...] = mean_val
      self.transformer.set_mean('data', channel_mean)
      self.transformer.set_channel_swap('data', (2, 1, 0))
      self.transformer.set_transpose('data', (2, 0, 1))
    else:
      print "Warning: did not set transformer; assume that image features do not need to be transformed.\n"
    # Setup sentence prediction net.
    self.lstm_nets = [caffe.Net(lstm_net_proto, weights_path, phase) for weights_path in weights_paths]
    if 'hidden_unit_in' in self.lstm_nets[0].blobs.keys():
      #SO HACKY!!!
      lstm_net_tmp = caffe.Net('mrnn_wtd_deploy_lm_im.prototxt', weights_path, phase)
      self.lstm_nets[0].params['lstm1_x_transform'][0].data[:] = copy.deepcopy(lstm_net_tmp.params['lstm1'][0].data)
      self.lstm_nets[0].params['lstm1_x_transform'][1].data[:] = copy.deepcopy(lstm_net_tmp.params['lstm1'][1].data)
      self.lstm_nets[0].params['lstm1_transform'][0].data[:] = copy.deepcopy(lstm_net_tmp.params['lstm1'][2].data)
      del lstm_net_tmp
    self.vocab = ['<EOS>']
    with open(vocab_path, 'r') as vocab_file:
      self.vocab += [word.strip() for word in vocab_file.readlines()]
    net_vocab_size = self.lstm_nets[0].blobs['predict-im'].data.shape[2]
    if len(self.vocab) != net_vocab_size:
      raise Exception('Invalid vocab file: contains %d words; '
          'net expects vocab with %d words' % (len(self.vocab), net_vocab_size))

    self.h5_file = precomputed_feats

  def set_image_batch_size(self, batch_size):
    self.image_net.blobs['data'].reshape(batch_size,
        *self.image_net.blobs['data'].data.shape[1:])



  def caption_batch_size(self):
    return self.lstm_nets[0].blobs['cont_sentence'].data.shape[1]

  def set_caption_batch_size(self, batch_size):
    dim = len(self.lstm_nets[0].blobs['image_features'].data.shape)
    for lstm_net in self.lstm_nets:
      lstm_net.blobs['cont_sentence'].reshape(1, batch_size)
      lstm_net.blobs['input_sentence'].reshape(1, batch_size)
      if 'hidden_unit_in' in lstm_net.blobs.keys():
        lstm_net.blobs['hidden_unit_in'].reshape(1, batch_size, lstm_net.blobs['hidden_unit_in'].data.shape[-1])
        lstm_net.blobs['cell_unit_in'].reshape(1, batch_size, lstm_net.blobs['cell_unit_in'].data.shape[-1])
  
      if dim == 2:
        lstm_net.blobs['image_features'].reshape(batch_size,
            *lstm_net.blobs['image_features'].data.shape[1:])
      elif dim == 3:
        lstm_net.blobs['image_features'].reshape(1, batch_size,
            *lstm_net.blobs['image_features'].data.shape[2:])
   
      lstm_net.reshape()

  def preprocess_image(self, image, verbose=False):
    if type(image) in (str, unicode):
      try:
        image = plt.imread(image)
      except: 
        print "Loading image with PIL"
        img = Image.open(str(image))
        b = type(np.array(img))
        a = np.array(img)
        del img
        img = a.astype(np.float32)
    crop_edge_ratio = (256. - 227.) / 256. / 2
    ch = int(image.shape[0] * crop_edge_ratio + 0.5)
    cw = int(image.shape[1] * crop_edge_ratio + 0.5)
    cropped_image = image[ch:-ch, cw:-cw]
    if len(cropped_image.shape) == 2:
      cropped_image = np.tile(cropped_image[:, :, np.newaxis], (1, 1, 3))
    preprocessed_image = self.transformer.preprocess('data', cropped_image)
    if verbose:
      print 'Preprocessed image has shape %s, range (%f, %f)' % \
          (preprocessed_image.shape,
           preprocessed_image.min(),
           preprocessed_image.max())
    return preprocessed_image

  def preprocessed_image_to_descriptor(self, image, output_name='fc8'):
    net = self.image_net
    if net.blobs['data'].data.shape[0] > 1:
      batch = np.zeros_like(net.blobs['data'].data)
      batch[0] = image[0]
    else:
      batch = image
    net.forward(data=batch)
    descriptor = net.blobs[output_name].data[0].copy()
    return descriptor

  def image_to_descriptor(self, image, output_name='fc8'):
    return self.preprocessed_image_to_descriptor(self.preprocess_image(image), output_name)

  def predict_single_word(self, descriptor, previous_word, output='probs', seed_net=''):
    nets = self.lstm_nets[0]
    if seed_net:
      net = seed_net #this is to ensure net maintains state for predict_missing word scheme
    cont = 0 if previous_word == 0 else 1
    cont_input = np.array([cont])
    word_input = np.array([previous_word])
    image_features = np.zeros_like(net.blobs['image_features'].data)
    image_features[:] = descriptor
    net.forward(image_features=image_features, cont_sentence=cont_input,
                input_sentence=word_input)
    output_preds = net.blobs[output].data[0, 0, :]
    return output_preds

  def predict_single_word_from_all_previous(self, descriptor, previous_words):
    for word in [0] + previous_words:
      probs = self.predict_single_word(descriptor, word)
    return probs

  def copy_net_blobs(self, net):
    copy_blobs = {}
    for key in net.blobs.keys():
      copy_blobs[key] = copy.deepcopy(net.blobs[key].data)
    return copy_blobs

  def reinitialize_blobs(self, copy_blobs, net):
    for key in net.blobs.keys():
      net.blobs[key].data[...] = copy.deepcopy(copy_blobs[key])
    return net

  def predict_missing_word_from_all_words_test(self, descriptor, previous_words, next_words, prob_output_name='probs'):
    descriptor = np.array(descriptor)
    #batch_size = len(self.vocab)
    batch_size = 5 
    nets = self.lstm_nets[0]
    outputs = []
    output_captions = [[] for b in range(batch_size)]
    log_probs = [[] for b in range(batch_size)]

    #prep the lstm with previous words  
    missing_word_probs = np.zeros((net.blobs[prob_output_name].data.shape[2],))
    for b in xrange(0,len(self.vocab), batch_size):  
      batch_end = min(len(self.vocab), b + batch_size)
      self.set_caption_batch_size(batch_size)
      cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
      word_input = np.zeros_like(net.blobs['input_sentence'].data)
      image_features = np.zeros_like(net.blobs['image_features'].data)
      image_features[:] = descriptor
      for wi, word in enumerate([0] + previous_words):
        if wi == 0:
          cont_input[:] = 0
        else:
          cont_input[:] = 1
        word_input[:] = word
        net.forward(image_features=image_features, cont_sentence=cont_input, input_sentence=word_input) 

      probs = net.blobs[prob_output_name].data[0]
      missing_word_probs[b:batch_end] = probs[0,b:batch_end] #p(word|prev_word)
      final_words = next_words + [0]

      self.set_caption_batch_size(batch_size)
      cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
      word_input = np.zeros_like(net.blobs['input_sentence'].data)
      image_features = np.zeros_like(net.blobs['image_features'].data)
      image_features[:] = descriptor

      #copy lstm_weights
      #seed_net_blobs = self.copy_net_blobs(self.lstm_net)  
      #now look at probabilities for placement of each vocab word
      #self.lstm_net = self.reinitialize_blobs(seed_net_blobs, self.lstm_net)
      #first propogate possible words to fill in blank
      cont_input[:] = 1
      word_input[0,:batch_end-b] = range(b, batch_end) 
      net.forward(image_features=image_features, cont_sentence=cont_input, input_sentence=word_input)
      missing_word_probs[b:batch_end] *= net.blobs[prob_output_name].data[0, 0:batch_end-b, final_words[0]] 
      for wi in range(len(final_words)-1):
        word_input[:] = final_words[wi]
        net.forward(image_features=image_features, cont_sentence=cont_input, input_sentence=word_input) 
        missing_word_probs[b:batch_end] *= net.blobs[prob_output_name].data[0, 0:batch_end-b, final_words[wi+1]] 
    
    missing_word_probs = missing_word_probs/np.sum(missing_word_probs) 
    return missing_word_probs

  def predict_missing_word_from_all_words(self, descriptor, previous_words, next_words, word_idx, beam_size, prob_output_name='probs'):
    #input: list of descriptors, previous_words next_words, word_idxs

    num_sentences = 1
    batch_size = beam_size*num_sentences
 
    descriptor = np.array(descriptor)
    nets = self.lstm_nets[0]
    outputs = []
    output_captions = [[] for b in range(batch_size)]

    #prep the lstm with previous words  

    batch_end = min(len(self.vocab), batch_size)
    self.set_caption_batch_size(batch_size)
    cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
    word_input = np.zeros_like(net.blobs['input_sentence'].data)
    image_features = np.zeros_like(net.blobs['image_features'].data)
    image_features[:] = descriptor
    previous_words = [0] + previous_words
    log_probs = np.zeros((batch_size,)) 

    for wi in range(len(previous_words)):
      if wi == 0:
        cont_input[:] = 0
      else:
        cont_input[:] = 1
      word_input[:] = previous_words[wi]
      net.forward(image_features=image_features, cont_sentence=cont_input, input_sentence=word_input) 
      if wi < len(previous_words)-1:
        log_probs[:] += np.log(net.blobs[prob_output_name].data[0,0,previous_words[wi+1]])

    probs = net.blobs[prob_output_name].data[0]

    track_idxs = np.argsort(probs[0,:])[-beam_size:]
    if not np.where(track_idxs == word_idx)[0]:
      track_idxs = np.append(track_idxs, word_idx)
      log_probs = np.append(log_probs, log_probs[-1])
      batch_size += 1
      batch_end = min(len(self.vocab), batch_size)

    log_probs += np.log(probs[0,track_idxs]) #p(word|prev_word)
    final_words = next_words + [0]

    self.set_caption_batch_size(batch_size)
    cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
    word_input = np.zeros_like(net.blobs['input_sentence'].data)
    image_features = np.zeros_like(net.blobs['image_features'].data)
    image_features[:] = descriptor

    cont_input[:] = 1
    word_input[0,:batch_end] = track_idxs 
    net.forward(image_features=image_features, cont_sentence=cont_input, input_sentence=word_input)
    log_probs[:batch_end] += np.log(net.blobs[prob_output_name].data[0, 0:batch_end, final_words[0]]) 
    for wi in range(len(final_words)-1):
      word_input[:] = final_words[wi]
      net.forward(image_features=image_features, cont_sentence=cont_input, input_sentence=word_input) 
      log_probs[:batch_end] += np.log(net.blobs[prob_output_name].data[0, 0:batch_end, final_words[wi+1]]) 
#    if np.sum(missing_word_probs) > 0:
#      missing_word_probs = missing_word_probs/np.sum(missing_word_probs)
    missing_word_probs_all = np.zeros((net.blobs[prob_output_name].data.shape[2],))
    for ix, idx in enumerate(track_idxs):
      missing_word_probs_all[idx] = log_probs[ix] 
    return missing_word_probs_all
       
  def fill_caption(self, descriptor, caption, word_idx, beam_size):
    num_sentences = len(caption)
    del_word = caption.index(word_idx)
#    previous_words = []
#    next_words = []
#    for i in range(num_sentences):
#      previous_words.append(caption[i][:del_word])
#      next_words.append(caption[i][del_word+1:])
    previous_words = caption[:del_word]
    next_words = caption[del_word+1:]
    return self.predict_missing_word_from_all_words(descriptor, previous_words, next_words, word_idx, beam_size)
 
  # Strategy must be either 'beam' or 'sample'.
  # If 'beam', do a max likelihood beam search with beam size num_samples.
  # Otherwise, sample with temperature temp.
  def predict_caption(self, descriptor, strategy={'type': 'beam'}):
    assert 'type' in strategy
    assert strategy['type'] in ('beam', 'sample')
    if strategy['type'] == 'beam':
      return self.predict_caption_beam_search(descriptor, strategy)
    num_samples = strategy['num'] if 'num' in strategy else 1
    samples = []
    sample_probs = []
    for _ in range(num_samples):
      sample, sample_prob = self.sample_caption(descriptor, strategy)
      samples.append(sample)
      sample_probs.append(sample_prob)
    return samples, sample_probs


  def sample_caption(self, descriptor, strategy,
                     net_output='predict', max_length=50):
    sentence = []
    probs = []
    eps_prob = 1e-8
    temp = strategy['temp'] if 'temp' in strategy else 1.0
    if max_length < 0: max_length = float('inf')
    while len(sentence) < max_length and (not sentence or sentence[-1] != 0):
      previous_word = sentence[-1] if sentence else 0
      softmax_inputs = self.predict_single_word(descriptor, previous_word,
                                                output=net_output)
      word = random_choice_from_probs(softmax_inputs, temp)
      sentence.append(word)
      probs.append(softmax(softmax_inputs, 1.0)[word])
    return sentence, probs

  def predict_caption_beam_search(self, descriptor, strategy, max_length=50):
    orig_batch_size = self.caption_batch_size()
    if orig_batch_size != 1: self.set_caption_batch_size(1)
    beam_size = strategy['beam_size'] if 'beam_size' in strategy else 1
    assert beam_size >= 1
    beams = [[]]
    beams_complete = 0
    beam_probs = [[]]
    beam_log_probs = [0.]
    while beams_complete < len(beams):
      expansions = []
      for beam_index, beam_log_prob, beam in \
          zip(range(len(beams)), beam_log_probs, beams):
        if beam:
          previous_word = beam[-1]
          if len(beam) >= max_length or previous_word == 0:
            exp = {'prefix_beam_index': beam_index, 'extension': [],
                   'prob_extension': [], 'log_prob': beam_log_prob}
            expansions.append(exp)
            # Don't expand this beam; it was already ended with an EOS,
            # or is the max length.
            continue
        else:
          previous_word = 0  # EOS is first word
        if beam_size == 1:
          probs = self.predict_single_word(descriptor, previous_word)
        else:
          probs = self.predict_single_word_from_all_previous(descriptor, beam)
        assert len(probs.shape) == 1
        assert probs.shape[0] == len(self.vocab)
        expansion_inds = probs.argsort()[-beam_size:]
        for ind in expansion_inds:
          prob = probs[ind]
          extended_beam_log_prob = beam_log_prob + math.log(prob)
          exp = {'prefix_beam_index': beam_index, 'extension': [ind],
                 'prob_extension': [prob], 'log_prob': extended_beam_log_prob}
          expansions.append(exp)
      # Sort expansions in decreasing order of probability.
      expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
      expansions = expansions[:beam_size]
      new_beams = \
          [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
      new_beam_probs = \
          [beam_probs[e['prefix_beam_index']] + e['prob_extension'] for e in expansions]
      beam_log_probs = [e['log_prob'] for e in expansions]
      beams_complete = 0
      for beam in new_beams:
        if beam[-1] == 0 or len(beam) >= max_length: beams_complete += 1
      beams, beam_probs = new_beams, new_beam_probs
    if orig_batch_size != 1: self.set_caption_batch_size(orig_batch_size)
    return beams, beam_probs

  def score_caption(self, descriptor, caption, is_gt=True, caption_source='gt'):
    output = {}
    output['caption'] = caption
    output['gt'] = is_gt
    output['source'] = caption_source
    output['prob'] = []
    probs = self.predict_single_word(descriptor, 0)
    for word in caption:
      output['prob'].append(probs[word])
      probs = self.predict_single_word(descriptor, word)
    return output

  def compute_descriptors(self, image_list, feats_bool=False, output_name='fc8'):
    coco=True
    if feats_bool:
      #load image from h5 file and make into a dict
      f = h5py.File(self.h5_file, 'r')
      extracted_features = {}
      for feature, im in zip(f['features'], f['ims']):
        if coco:
          im_key = im.split('_')[-1].split('.jpg')[0]
        else:
          im_key = im
        extracted_features[im_key] = feature

    batch = np.zeros_like(self.image_net.blobs['data'].data)
    batch_shape = batch.shape
    batch_size = batch_shape[0]
    descriptors_shape = (len(image_list), ) + \
        (self.image_net.blobs[output_name].data.shape[-1], )
    descriptors = np.zeros(descriptors_shape)
    for batch_start_index in range(0, len(image_list), batch_size):
      batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
      for batch_index, image_path in enumerate(batch_list):
        if feats_bool:
          if coco:
            im_key = image_path.split('_')[-1].split('.jpg')[0]
          else:
            im_key = im
          batch[batch_index:(batch_index+1),:] = extracted_features[im_key] 
        else:
          batch[batch_index:(batch_index + 1)] = self.preprocess_image(image_path)
      current_batch_size = min(batch_size, len(image_list) - batch_start_index)
      print 'Computing descriptors for images %d-%d of %d' % \
          (batch_start_index, batch_start_index + current_batch_size - 1,
           len(image_list))
      self.image_net.forward(data=batch)
      descriptors[batch_start_index:(batch_start_index + current_batch_size)] = \
          self.image_net.blobs[output_name].data[:current_batch_size]
    return descriptors

  def score_captions(self, descriptor, captions,
                     output_name='probs', caption_source='gt', verbose=True):
    net = self.lstm_nets[0]
    cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
    word_input = np.zeros_like(net.blobs['input_sentence'].data)
    image_features = np.zeros_like(net.blobs['image_features'].data)
    batch_size = image_features.shape[0]
    assert descriptor.shape == image_features.shape[1:]
    for index in range(batch_size):
      image_features[index] = descriptor
    outputs = []
    input_data_initialized = False
    for batch_start_index in range(0, len(captions), batch_size):
      caption_batch = captions[batch_start_index:(batch_start_index + batch_size)]
      current_batch_size = len(caption_batch)
      caption_index = 0
      probs_batch = [[] for b in range(current_batch_size)]
      num_done = 0
      while num_done < current_batch_size:
        if caption_index == 0:
          cont_input[:] = 0
        elif caption_index == 1:
          cont_input[:] = 1
        for index, caption in enumerate(caption_batch):
          word_input[0, index] = \
              caption['caption'][caption_index - 1] if \
              0 < caption_index < len(caption['caption']) else 0
        if input_data_initialized:
          net.forward(start="embedding", input_sentence=word_input,
                      cont_sentence=cont_input, image_features=image_features)
        else:
          net.forward(input_sentence=word_input, cont_sentence=cont_input,
                      image_features=image_features)
          input_data_initialized = True
        output_probs = net.blobs[output_name].data
        for index, probs, caption in \
            zip(range(current_batch_size), probs_batch, caption_batch):
          if caption_index == len(caption['caption']) - 1:
            num_done += 1
          if caption_index < len(caption['caption']):
            word = caption['caption'][caption_index]
            probs.append(output_probs[0, index, word].reshape(-1)[0])
        if verbose:
          print 'Computed probs for word %d of captions %d-%d (%d done)' % \
              (caption_index, batch_start_index,
               batch_start_index + current_batch_size - 1, num_done)
        caption_index += 1
      for prob, caption in zip(probs_batch, caption_batch):
        output = {}
        output['caption'] = caption['caption']
        output['prob'] = prob
        output['gt'] = True
        output['source'] = caption_source
        outputs.append(output)
    return outputs
 
  def init_zeros(self, descriptor):
    hidden_unit_in = np.zeros_like(self.lstm_net.blobs['hidden_unit_in'].data)
    cell_unit_in = np.zeros_like(self.lstm_net.blobs['cell_unit_in'].data)
    return hidden_unit_in, cell_unit_in

  def init_average(self, descriptor):
     d_reshape = descriptor.reshape((descriptor.shape[0], 512, 28,28))
     feat_average = np.sum(d_reshape, axis=3)
     feat_average = np.sum(feat_average, axis=2)
     hidden_unit_W = self.lstm_net.params['HiddenUnitInitialization'][0].data
     hidden_unit_b = self.lstm_net.params['HiddenUnitInitialization'][1].data
     cell_unit_W = self.lstm_net.params['CellInitialization'][0].data
     cell_unit_b = self.lstm_net.params['CellInitialization'][1].data
     hidden_unit_in = np.dot(feat_average, hidden_unit_W.T) + np.tile(hidden_unit_b, (feat_average.shape[0],1)) 
     cell_unit_in = np.dot(feat_average, cell_unit_W.T) + np.tile(cell_unit_b, (feat_average.shape[0],1)) 
     return hidden_unit_in, cell_unit_in
 
  def sample_captions(self, descriptor, prob_output_name='probs',
                      pred_output_name='predict', temp=1, max_length=50, 
                      min_length=2, hidden_init=None, forward_model = '', descriptor_filename=''):
    descriptor = np.array(descriptor)
    batch_size = descriptor.shape[0]
    self.set_caption_batch_size(batch_size)
    nets = self.lstm_nets
    prev_word_bool = self.prev_word_bool

    cont_input = np.zeros_like(nets[0].blobs['cont_sentence'].data)
    word_input = np.zeros_like(nets[0].blobs['input_sentence'].data)
    image_features = np.zeros_like(nets[0].blobs['image_features'].data)
    if 'hidden_unit_in' in nets[0].blobs.keys():
      #hidden_unit_in, cell_unit_in = self.init_average(descriptor)
      hidden_unit_in, cell_unit_in = self.init_zeros(descriptor)   

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
      if forward_model == 'sat_rough':
        t = caption_index + 1
        print t
        if t == 1:
          nets[0].forward(image_features=image_features, cont_sentence=cont_input,
                      input_sentence=word_input, end="LSTMUnit_1")
        else:
          nets[0].forward(image_features=image_features, cont_sentence=cont_input,
                      input_sentence=word_input, end="wtSplit")
          nets[0].forward(start="HiddenAttentionTransform_"+str(t-1), end="LSTMUnit_"+str(t))
        nets[0].blobs['reshape_z_20'].data[:] = net.blobs['reshape_z_'+str(t)].data
        nets[0].blobs['hidden_unit_20'].data[:] = net.blobs['hidden_unit_'+str(t)].data
        nets[0].forward(start="ConcatVisualWordFeatures")

      elif not 'hidden_unit_in' in nets[0].blobs.keys():
        #t = time.time()
        net_output_probs = np.zeros((nets[0].blobs[prob_output_name].data[0].shape))
        for net in nets:
          net.forward(image_features=image_features, cont_sentence=cont_input,
                    input_sentence=word_input)
          net_output_probs += net.blobs[prob_output_name].data[0]
        #print "time for forward is %f\n" %(time.time()-t)
        #print 'stop point'
      else:
        nets[0].forward(image_features=image_features, cont_sentence=cont_input,
                    input_sentence=word_input, hidden_unit_in=hidden_unit_in, cell_unit_in=cell_unit_in)
        hidden_unit_in[:] = net.blobs['hidden_unit_out'].data 
        cell_unit_in[:] = net.blobs['cell_unit_out'].data 
      if temp == 1.0 or temp == float('inf'):
        #t = time.time()
        #when i changed the lm_net to input lists I moved the following line; this might cause issues
        #net_output_probs = net.blobs[prob_output_name].data[0]
        no_EOS = False if (caption_index > min_length) else True
        prev_words = [False]*len(output_captions)
        if prev_word_bool & caption_index > 0:
          prev_words = [output_caption[-1] for output_caption in output_captions]
        samples = [
            random_choice_from_probs(dist, temp=temp, already_softmaxed=True, no_EOS=no_EOS, prev_word=prev_word)
            for dist, prev_word in zip(net_output_probs, prev_words)
        ]
        #print "Time for determining next word is %f\n" %(time.time()-t)
      else:
        #net_output_preds = net.blobs[pred_output_name].data[0]
        samples = [
            random_choice_from_probs(preds, temp=temp, already_softmaxed=False)
            for preds in net_output_preds
        ]
      for index, next_word_sample in enumerate(samples):
        # If the caption is empty, or non-empty but the last word isn't EOS,
        # predict another word.
        if not output_captions[index] or output_captions[index][-1] != 0:
          output_captions[index].append(next_word_sample)
          output_probs[index].append(net_output_probs[index, next_word_sample])
          if next_word_sample == 0: num_done += 1
      sys.stdout.write('\r%d/%d done after word %d' %
          (num_done, batch_size, caption_index))
      sys.stdout.flush()
      caption_index += 1
    sys.stdout.write('\n')
    print caption_index
    return output_captions, output_probs

  def sentence(self, vocab_indices):
    if not vocab_indices:
      print "Had issue with image!"
      return ' '
    sentence = ' '.join([self.vocab[i] for i in vocab_indices])
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

def softmax(softmax_inputs, temp):
  shifted_inputs = softmax_inputs - softmax_inputs.max()
  exp_outputs = np.exp(temp * shifted_inputs)
  exp_outputs_sum = exp_outputs.sum()
  if math.isnan(exp_outputs_sum):
    return exp_outputs * float('nan')
  assert exp_outputs_sum > 0
  if math.isinf(exp_outputs_sum):
    return np.zeros_like(exp_outputs)
  eps_sum = 1e-20
  return exp_outputs / max(exp_outputs_sum, eps_sum)

def random_choice_from_probs(softmax_inputs, temp=1, already_softmaxed=False, no_EOS=False, prev_word = None):
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

def gen_stats(prob, normalizer=None):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += math.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  stats['p'] = math.exp(stats['log_p'])
  stats['p_word'] = math.exp(stats['log_p'])
  try:
    stats['perplex'] = math.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = math.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  if normalizer is not None:
    norm_stats = gen_stats(normalizer)
    stats['normed_perplex'] = stats['perplex'] / norm_stats['perplex']
    stats['normed_perplex_word'] = \
        stats['perplex_word'] / norm_stats['perplex_word']
  return stats
