#separates captions and images based on coco annotations as opposed to sentence captions

import h5py
import sys
import random
import json
import os
import pickle as pkl
import re

COCO_PATH = '../../data/coco/coco/'
feature_dir = '/y/lisaanne/image_captioning/coco_features/'
save_dir = 'h5_data/buffer_100/'
coco_anno_path = '%s/annotations/captions_%%s2014.json' % COCO_PATH
coco_txt_path = '/home/lisaanne/caffe-LSTM/data/coco/coco2014_cocoid.%s.txt'
random.seed(10)

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file) 

def init(json_dict):
  #initialize json captions dict
  json_dict['info'] = train_captions['info'] 
  json_dict['licenses'] = train_captions['info'] 
  json_dict['type'] = train_captions['type']
  json_dict['annotations'] = [] 
  json_dict['images'] = []
  return json_dict  

def save_files(dump_json, identifier):
  file_save = coco_anno_path %(identifier)
  txt_save = coco_txt_path %(identifier)
  with open(file_save, 'w') as outfile:
    json.dump(dump_json, outfile)
  write_txt_file(txt_save, identifier, dump_json)
  print 'Wrote json to %s.\n' %file_save
  print 'Wrote image txt to %s.\n' %txt_save

def coco_segmentation_label(image):
  #determine labels for given image
  return pass

def parse_label(image):
  #determine labels for given image
  return pass

def split_captions(json_dict, rm_word=None, label_filters=[coco_segmentation_label]):
  #split json captions into sentences which do not include rm_word and which do include rm_word
  json_dict_rm_word = {} #dict with word removed
  json_dict_rm_word = init(json_dict_rm_word)

  image_ids = [json_dict['annotations'][i]['image_id'] for i in range(len(json_dict['annotations']))]
  novel_annotation_ids = [] 
  novel_image_ids = [] 
  train_annotation_ids = [] 
  train_image_ids = [] 

  for count_im, im in enumerate(val_dict['images']):
    if count_im % 50 == 0:
      sys.stdout.write("\rAdding sentences for im %d/%d" % (count_im, len(val_dict['images'])))
      sys.stdout.flush()

    im_id = im['id']
    anno_idxs = [ix for ix, image_id in enumerate(image_ids) if anno_id == im_id]
    im_annotations = [json_dict['annotations'][anno_idx]['caption'] for anno_idx in anno_idxs]

    labels = []
    for label_filter in label_fitlers:
      labels.extend(label_filter(im_annotations))
 
    if rm_word:
      if rm_word in labels:
        rm_ids.extend(anno_idxs)
	rm_image_ids.append(count_im)

  if rm_word:
    for rm_id in sorted(rm_ids)[::-1]:
      a = json_dict['annotations'].pop(rm_id)
      json_dict_rm_word['annotations'].append(a)
    for rm_image_id in sorted(rm_image_ids)[::-1]:
      a = json_dict['images'].pop(rm_image_id)
      json_dict_rm_word['images'].append(a)

  return json_dict, json_dict_rm_word



    

