#!/usr/bin/env python

import pdb
import sys
sys.path.insert(0,'../')
from init import *
sys.path.insert(0,caffe_dir)
sys.path.insert(0,caffe_dir + '/python/')
from collections import OrderedDict
import json
import numpy as np
import pprint
import cPickle as pickle
import string
import argparse
import glob
sys.path.insert(0, '.')

# seed the RNG so we evaluate on the same subset each time
np.random.seed(seed=0)

from coco_to_hdf5_data import *
from captioner import Captioner

import caffe
COCO_EVAL_PATH = caffe_dir + '/data/coco/coco-caption-eval/'
sys.path.insert(0,COCO_EVAL_PATH)
from pycocoevalcap.eval import COCOEvalCap

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

class CaptionExperiment():
  # captioner is an initialized Captioner (captioner.py)
  # dataset is a dict: image path -> [caption1, caption2, ...]
  def __init__(self, captioner=None, dataset={}, dataset_cache_dir=None, cache_dir=None, sg=None, feats_bool=True, output_name='fc8'):
    self.captioner = captioner
    if not self.captioner:
      print "Warning: Have not selected captioner"
    self.sg = sg
    if not self.sg:
      print "Warning: Have not selected sequence generator"
    self.dataset_cache_dir = dataset_cache_dir
    self.cache_dir = cache_dir
    if (dataset_cache_dir) and (cache_dir):
      for d in [dataset_cache_dir, cache_dir]:
        if not os.path.exists(d): os.makedirs(d)
    self.dataset = dataset
    self.images = dataset.keys()
    self.init_caption_list(dataset)
    self.caption_scores = [None] * len(self.images)
    print 'Initialized caption experiment: %d images, %d captions' % \
        (len(self.images), len(self.captions))
    self.output_name = output_name

  def init_caption_list(self, dataset):
    self.captions = []
    for image, captions in dataset.iteritems():
      for caption, _ in captions:
        self.captions.append({'source_image': image, 'caption': caption})
    # Sort by length for performance.
    self.captions.sort(key=lambda c: len(c['caption']))

  def compute_num_descriptor_files(self):
    num_images = len(self.images) 
    shape_descriptor = (len(self.images), ) + \
                       (self.captioner.image_net.blobs[self.output_name].data.shape[-1], )     
    size_descriptor = 1
    for s in shape_descriptor: size_descriptor*= s
    max_size = 1000*50000.
    return int(np.ceil(size_descriptor/max_size))
      
  def compute_descriptors(self, des_file_idx=0, file_load=True):
    descriptor_filename = '%s/descriptors_%d.npz' % (self.dataset_cache_dir, des_file_idx)
    if os.path.exists(descriptor_filename) & file_load:
      self.descriptors = np.load(descriptor_filename)['descriptors']
      self.descriptor_filename=np.load(descriptor_filename)['image_id_array']
    else:
      num_des_files = self.compute_num_descriptor_files()
      start_image = (len(self.images)/num_des_files)*des_file_idx
      end_image = min((len(self.images)/num_des_files)*(des_file_idx+1), len(self.images)) 
      self.descriptors = self.captioner.compute_descriptors(self.images[start_image:end_image], output_name=self.output_name)
      image_id_array = [None]*len(self.images) 
      for i, im in enumerate(self.images):
        image_id_array[i] = im 
      self.descriptor_filename=image_id_array
      np.savez_compressed(descriptor_filename, descriptors=self.descriptors, image_id_array=image_id_array)

  def score_captions(self, image_index, output_name='probs'):
    assert image_index < len(self.images)
    caption_scores_dir = '%s/caption_scores' % self.cache_dir
    if not os.path.exists(caption_scores_dir):
      os.makedirs(caption_scores_dir)
    caption_scores_filename = '%s/scores_image_%06d.pkl' % \
        (caption_scores_dir, image_index)
    if os.path.exists(caption_scores_filename):
      with open(caption_scores_filename, 'rb') as caption_scores_file:
        outputs = pickle.load(caption_scores_file)
    else:
      outputs = self.captioner.score_captions(self.descriptors[image_index],
          self.captions, output_name=self.output_name, caption_source='gt',
          verbose=False)
      self.caption_stats(image_index, outputs)
      with open(caption_scores_filename, 'wb') as caption_scores_file:
        pickle.dump(outputs, caption_scores_file)
    self.caption_scores[image_index] = outputs

  def caption_stats(self, image_index, caption_scores):
    image_path = self.images[image_index]
    for caption, score in zip(self.captions, caption_scores):
      assert caption['caption'] == score['caption']
      score['stats'] = gen_stats(score['prob'])
      score['correct'] = (image_path == caption['source_image'])

  def score_generation(self, json_filename=None, generation_result=None):
    if not generation_result:
      generation_result = self.sg.coco.loadRes(json_filename)
      coco_dict = read_json(json_filename)
    coco_evaluator = COCOEvalCap(self.sg.coco, generation_result)
    #coco_image_ids = [self.sg.image_path_to_id[image_path]
    #                  for image_path in self.images]
    coco_image_ids = [j['image_id'] for j in coco_dict]
    coco_evaluator.params['image_id'] = coco_image_ids
    results = coco_evaluator.evaluate(return_results=True)
    return results

  def generate_captions(self, strategy, do_batches, batch_size, image_index=0):
    num_descriptors = self.descriptors.shape[0]
    all_captions = [None] * num_descriptors

    # Generate captions for all images.
    for descriptor_index in xrange(0, num_descriptors, batch_size):
      batch_end_index = min(descriptor_index + batch_size, num_descriptors)
      sys.stdout.write("\rGenerating captions for image %d/%d" %
                       (image_index, num_descriptors))
      sys.stdout.flush()
      if do_batches:
        if strategy['type'] == 'beam' or \
            ('temp' in strategy and strategy['temp'] == float('inf')):
          temp = float('inf')
        else:
          temp = strategy['temp'] if 'temp' in strategy else 1
        output_captions, output_probs = self.captioner.sample_captions(
            self.descriptors[descriptor_index:batch_end_index], temp=temp, min_length = 2, descriptor_filename=self.descriptor_filename[descriptor_index:batch_end_index])
        for batch_index, output in zip(range(descriptor_index, batch_end_index),
                                       output_captions):
          all_captions[image_index] = output
          image_index += 1
      else:
        for batch_image_index in xrange(descriptor_index, batch_end_index):
          captions, caption_probs = self.captioner.predict_caption(
              self.descriptors[batch_image_index], strategy=strategy)
          best_caption, max_log_prob = None, None
          for caption, probs in zip(captions, caption_probs):
            log_prob = gen_stats(probs)['log_p']
            if best_caption is None or \
                (best_caption is not None and log_prob > max_log_prob):
              best_caption, max_log_prob = caption, log_prob
          all_captions[image_index] = best_caption
          image_index += 1
    sys.stdout.write('\n')
    return all_captions, image_index
  
  def save_and_score_generation(self, all_captions):
    # Compute the number of reference files as the maximum number of ground
    # truth captions of any image in the dataset.
    num_reference_files = 0
    for captions in self.dataset.values():
      if len(captions) > num_reference_files:
        num_reference_files = len(captions)
    if num_reference_files <= 0:
      raise Exception('No reference captions.')

    # Collect model/reference captions, formatting the model's captions and
    # each set of reference captions as a list of len(self.images) strings.
    exp_dir = '%s/generation' % self.cache_dir
    if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
    # For each image, write out the highest probability caption.
    model_captions = [''] * len(self.images)
    reference_captions = [([''] * len(self.images)) for _ in xrange(num_reference_files)]
    for image_index, image in enumerate(self.images):
      caption = self.captioner.sentence(all_captions[image_index])
      model_captions[image_index] = caption
      for reference_index, (_, caption) in enumerate(self.dataset[image]):
        caption = ' '.join(caption)
        reference_captions[reference_index][image_index] = caption

    coco_image_ids = [self.sg.image_path_to_id[image_path]
                      for image_path in self.images]
    generation_result = [{
      'image_id': self.sg.image_path_to_id[image_path],
      'caption': model_captions[image_index]
    } for (image_index, image_path) in enumerate(self.images)]
    json_filename = '%s/generation_result.json' % self.cache_dir
    print 'Dumping result to file: %s' % json_filename
    with open(json_filename, 'w') as json_file:
      json.dump(generation_result, json_file)
    generation_result = self.sg.coco.loadRes(json_filename)
    coco_evaluator = COCOEvalCap(self.sg.coco, generation_result)
    coco_evaluator.params['image_id'] = coco_image_ids
    coco_evaluator.evaluate()

  def generation_experiment(self, strategy, max_batch_size=1000):
    # Compute image descriptors.

    if self.captioner.image_net:
      num_des_files = self.compute_num_descriptor_files()
    #for i in range(0, num_des_files):
    #  print 'Computing image descriptors (%d/%d)' %(i, num_des_files)
    #  self.compute_descriptors(i, file_load=False)
     
    num_images = len(self.images)
    do_batches = (strategy['type'] == 'beam' and strategy['beam_size'] == 1) or \
        (strategy['type'] == 'sample' and
         ('temp' not in strategy or strategy['temp'] in (1, float('inf'))) and
         ('num' not in strategy or strategy['num'] == 1))
    batch_size = min(max_batch_size, num_images) if do_batches else 1
    all_captions = [None] * num_images
    image_index = 0
 
    for i in range(0, num_des_files):

      print 'Computing image descriptors'
      self.compute_descriptors(i)
      all_captions[i:self.descriptors.shape[0]], image_index = self.generate_captions(strategy, do_batches, batch_size, image_index=image_index)

    self.save_and_score_generation(all_captions)

def gen_stats(prob):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += np.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  try:
    stats['perplex'] = np.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = np.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  return stats

def determine_anno_path(dataset_name, split_name):
  if dataset_name == 'coco':
    return COCO_ANNO_PATH % split_name
  if dataset_name == 'birds':
    return bird_anno_path % split_name 
  if dataset_name == 'birds_fg':
    return bird_anno_path_fg % (split_name, split_name)
  else:
    raise Exception ("do not annotation path for dataset %s." %dataset_name)

def determine_image_pattern(dataset_name, split_name):
  if dataset_name == 'coco':
    split_name_hash = {'train': 'train', 'val': 'val', 'val_val': 'val', 'val_test': 'val'}
    return COCO_IMAGE_PATTERN %(split_name_hash[split_name])   
  if dataset_name == 'birds':
    return '/yy2/lisaanne/fine_grained/CUB_200_2011/images/'
  if dataset_name == 'birds_fg':
    return '/yy2/lisaanne/fine_grained/CUB_200_2011/images/'
  else:
    raise Exception ("do not know image pattern for dataset %s." %dataset_name) 

def determine_vocab_folder(dataset_name, split_name):
  if dataset_name == 'birds':
    return bird_vocab_path 
  if dataset_name == 'birds_fg':
    return bird_vocab_path 
  else:
    raise Exception ("do not know vocab folder for dataset %s." %dataset_name) 

def build_sequence_generator(anno_path, buffer_size, image_root, vocab, max_words, align=False, shuffle=False, gt_captions=True, pad=True, truncate=True, split_ids=None):
  coco = COCO(anno_path)
  return CocoSequenceGenerator(coco, buffer_size, image_root, vocab, max_words, align, shuffle, gt_captions, pad, truncate, split_ids) 

def build_captioner(model_name, image_net, LM_net, dataset_name='coco', split_name='val', vocab='vocabulary', precomputed_h5=None, gpu=0, prev_word_restriction=True):
  model_files = ['%s.caffemodel' % (mf) for mf in model_name] 
  if image_net:
    image_net_file = home_dir + image_net 
  else: 
    image_net_file = None
  lstm_net_file = home_dir + LM_net
  vocab_file = '%s/%s.txt' %(determine_vocab_folder(dataset_name, split_name), vocab)
  device_id = gpu
  with open(vocab_file, 'r') as vocab_file_read:
    vocab = [line.strip() for line in vocab_file_read.readlines()]
  anno_path = determine_anno_path(dataset_name, split_name)
  image_root = determine_image_pattern(dataset_name, split_name) 

  sg = build_sequence_generator(anno_path, BUFFER_SIZE, image_root, vocab=vocab, 
                            max_words=MAX_WORDS, align=False, shuffle=False,
                            gt_captions=True, pad=True, truncate=True, 
                            split_ids=None)
  dataset = {}
  for image_path, sentence in sg.image_sentence_pairs:
    if image_path not in dataset:
      dataset[image_path] = []
    dataset[image_path].append((sg.line_to_stream(sentence), sentence))
  print 'Original dataset contains %d images' % len(dataset.keys())
  captioner = Captioner(model_files, image_net_file, lstm_net_file, vocab_file,
                        device_id=device_id, precomputed_feats=precomputed_h5, prev_word_restriction=prev_word_restriction)
  return captioner, sg, dataset

def main(model_name,image_net, LM_net,  dataset_name='coco', split_name='val', vocab='vocabulary', precomputed_h5=None, experiment={'type': 'generation', 'prev_word_restriction': False}, gpu=0):
  #model_name is the trained model: path relative to /home/lisa/caffe-LSTM-video
  #image_net is the model to extractimage_features 
  #dataset_name indicates which dataset to look at
  #split_name indicates which split of the dataset to look at
  #vocab indicates which vocabulary file to look at
  #experiment: dict which has all info needed for experiments including type and strategy   

  if not 'prev_word_restriction' in experiment.keys():
    experiment['prev_word_restriction'] = False

  captioner, sg, dataset = build_captioner(model_name, image_net, LM_net, dataset_name, split_name, vocab, precomputed_h5, gpu, experiment['prev_word_restriction']) 

  if 'beam_size' in experiment.keys():
    beam_size = experiment['beam_size']
  else:
    beam_size = 1
  generation_strategy = {'type': 'beam', 'beam_size': beam_size}
  if generation_strategy['type'] == 'beam':
    strategy_name = 'beam%d' % generation_strategy['beam_size']
  elif generation_strategy['type'] == 'sample':
    strategy_name = 'sample%f' % generation_strategy['temp']
  else:
    raise Exception('Unknown generation strategy type: %s' % generation_strategy['type'])

  dataset_subdir = '%s_%s' % (dataset_name, split_name)
  dataset_cache_dir = '%s/%s/%s' % (cache_home, dataset_subdir, model_name[0])
  feature_cache_dir = '%s/%s/%s' % (cache_home, dataset_subdir, model_name[0])
  cache_dir = '%s/%s' % (dataset_cache_dir, strategy_name)
  experimenter = CaptionExperiment(captioner, dataset, feature_cache_dir, cache_dir, sg)
  captioner.set_image_batch_size(min(100, len(dataset.keys())))
  if experiment['type'] == 'generation':
    experimenter.generation_experiment(generation_strategy, 1000)
  if experiment['type'] == 'score_generation':
    if 'read_file' in experiment.keys(): read_file=experiment['read_file']
    else: read_file=True
    experimenter.score_generation(json_filename=experiment['json_file'])

