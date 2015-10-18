import h5py
import sys
import random
import json
import os
import pickle as pkl
import re
import time

COCO_PATH = '../../data/coco/coco/'
feature_dir = '/y/lisaanne/image_captioning/coco_features/'
save_dir = 'h5_data/buffer_100/'
coco_anno_path = '%s/annotations/captions_%%s2014.json' % COCO_PATH
coco_instance_path = '%s/annotations/instances_%%s2014.json' % COCO_PATH
coco_txt_path = '/home/lisaanne/caffe-LSTM/data/coco/coco2014_cocoid.%s.txt'
random.seed(10)
label_filter_split = 'parse_label'

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file) 

def split_sent(sent):
  re.sub('[^A-Za-z0-9\s]+','',sent)
  return sent.split(' ')

#load relevant json dicts
instances_train = read_json(coco_instance_path % 'train')
instances_val = read_json(coco_instance_path % 'val')
anno_train = read_json(coco_anno_path %'train')
anno_val_val = read_json(coco_anno_path %'val_val')
anno_val_test = read_json(coco_anno_path %'val_test')

segmentation_category_dict = {}
for c in instances_train['categories']:
  segmentation_category_dict[c['id']] = c['name']


if label_filter_split == 'coco_segmentation_label':
  t = time.time()
  instances_train_dict = {}
  for ix, a in enumerate(instances_train['annotations']):
    sys.stdout.write("\rCreating instances train dict %d/%d" % (ix, len(instances_train['annotations'])))
    sys.stdout.flush()
    if a['id'] in instances_train_dict.keys():
      instances_train_dict[a['id']].append(a['category_id'])
    else:
      instances_train_dict[a['id']] = []
      
  print "Creating instances_train_dict took: %s s.\n" %time.time()-t

  t = time.time()
  instances_val_dict = {}
  for ix, a in enumerate(instances_val['annotations']):
    sys.stdout.write("\rCreating instances train val %d/%d" % (ix, len(instances_train['annotations'])))
    sys.stdout.flush()
    if a['id'] in instances_val_dict.keys():
      instances_val_dict[a['id']].append(a['category_id'])
    else:
      instances_val_dict[a['id']] = []
      
  print "Creating instances_val_dict took: %s s.\n" %time.time()-t

def init(json_dict):
  #initialize json captions dict
  json_dict['info'] = anno_train['info'] 
  json_dict['licenses'] = anno_train['info'] 
  json_dict['type'] = anno_train['type']
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

def coco_segmentation_label(im_id, im_annotations, train_or_val):
  #determine labels for given image
  if train_or_val == 'train': instances_dict = instance_train_dict
  if train_or_val == 'val': instances_dict = instance_val_dict

  instances = instances_dict[im_id]
  instance_names = [segmentation_category_dict[i] for i in instances]
  return set(instance_names)

def parse_label(im_id, im_annotations, train_or_val=None):
  #determine labels for given image
  words = [] 
  for anno in im_annotations:
    words.append(split_sent(anno))
  return set(words)

def clean_captions(json_dict, rm_wrd):
  #take captions out if they contain certain words
  bad_captions_idx = [i for i, caption in enumerate(json_dict['annotations']) if rm_word in split_sent(caption)]
  for bad_idx in sorted(bad_captions_idx)[::-1]:
    json_dict['annotations'].pop(bad_idx)

  return json_dict 

def save_files(dump_json, identifier):
  file_save = coco_anno_path %(identifier)
  txt_save = coco_txt_path %(identifier)
  with open(file_save, 'w') as outfile:
    json.dump(dump_json, outfile)
  write_txt_file(txt_save, identifier, dump_json)
  print 'Wrote json to %s.\n' %file_save
  print 'Wrote image txt to %s.\n' %txt_save

def split_captions(json_dict, rm_words=None, label_filter=coco_segmentation_label, train_or_val='train'):
  #split json captions into sentences which do not include rm_word and which do include rm_word
  json_dict_rm_word = {} #dict with word removed
  json_dict_rm_word = init(json_dict_rm_word)

  image_ids = [json_dict['annotations'][i]['image_id'] for i in range(len(json_dict['annotations']))]
  novel_annotation_ids = [] 
  novel_image_ids = [] 
  train_annotation_ids = [] 
  train_image_ids = [] 

  for count_im, im in enumerate(json_dict['images']):
    if count_im % 50 == 0:
      sys.stdout.write("\rAdding sentences for im %d/%d" % (count_im, len(json_dict['images'])))
      sys.stdout.flush()

    im_id = im['id']
    anno_idxs = [ix for ix, annotations in enumerate(json_dict['annotations']) if annotations['image_id'] == im_id]
    im_annotations = [json_dict['annotations'][anno_idx]['caption'] for anno_idx in anno_idxs]

    labels = label_filter(im_id, im_annotations, train_or_val)

    rm_ids = []
    rm_image_ids = []
 
    if rm_words:
      if len(set.intersection(set(rm_words),labels)) > 0:
        rm_ids.extend(anno_idxs)
	rm_image_ids.append(count_im)

  if rm_words:
    for rm_id in sorted(rm_ids)[::-1]:
      a = json_dict['annotations'].pop(rm_id)
      json_dict_rm_word['annotations'].append(a)
    for rm_image_id in sorted(rm_image_ids)[::-1]:
      a = json_dict['images'].pop(rm_image_id)
      json_dict_rm_word['images'].append(a)

  return json_dict, json_dict_rm_word

split_word_lists = [['zebra']]
rm_sentence_words = [[]]
save_tags = ['zebra']

if label_filter_split == 'coco_segmentation_label':
  label_filter_split = coco_segmentation_label
if label_filter_split == 'parse_label':
  label_filter_split = parse_label

for split_word_list, rm_sentence_word, save_tag in zip(split_word_lists, rm_sentence_words, save_tags):
  train_novel, train_train = split_captions(anno_train, rm_words=split_word_list, label_filter=label_filter_split, train_or_val='train')
  val_val_novel, val_val_train = split_captions(anno_val_val, rm_words=split_word_list, label_filter=label_filter_split, train_or_val='val')
  val_test_novel, val_test_train = split_captions(anno_val_test, rm_words=split_word_list, label_filter=label_filter_split, train_or_val='val')
# I decided not to do this; but this should clean the capions and make sure the removed word is not accidentally used in the captions (people frequently switch zebra and giraffe)
#  for rm_word in rm_sentence_word:
#    train_train = clean_captions(train_train, rm_word)
#    val_val_novel = clean_captions(val_val_novel, rm_word)
#    val_val_train = clean_captions(val_val_train, rm_word)
#    val_test_novel = clean_captions(val_test_novel, rm_word)
#    val_test_train = clean_captions(val_test_train, rm_word)

  save_files(train_train, save_tag+'_ss_train')
  save_files(val_val_novel, save_tag+'_ss_val_val_novel')
  save_files(val_val_train, save_tag+'_ss_val_val_train')
  save_files(val_val_novel, save_tag+'_ss_val_val_novel')
  save_files(val_val_train, save_tag+'_ss_val_val_train')
    

