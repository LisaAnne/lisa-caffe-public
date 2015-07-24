import h5py
import sys
import random
import json
import os

COCO_PATH = '/y/lisaanne/coco'
feature_dir = '/y/lisaanne/image_captioning/coco_features/'
save_dir = 'h5_data/buffer_100/'
coco_anno_path = '%s/annotations/captions_%%s2014.json' % COCO_PATH
coco_txt_path = '/home/lisa/caffe-LSTM-video/data/coco/coco2014_cocoid.%s.txt'
random.seed(10)

def create_hdf5(split_name):
  files_txt = open(split_name, 'rb')
  files = files_txt.readlines()
  
def combine_json_dict(current_dict, add_json, json_name):
  for ix, image in enumerate(add_json['images']):
    image_id = image['id']
    if image_id in add_json.keys():
      print 'There is an issue!  This image_id has already been added to current dict.\n'
    current_dict[image_id] = {}
    current_dict[image_id]['image'] = image['file_name']
    current_dict[image_id]['json'] = json_name
    current_dict[image_id]['list_index'] = ix
  for ix, a in enumerate(add_json['annotations']):
    if ix % 100 == 0:
      print 'On annotation %d of %d.\n' %(ix, len(add_json['annotations']))
    caption = a['caption']  
    image_id = a['image_id']
    if image_id not in current_dict.keys():
      print "There is an issue!  Annotation image id is NOT in current dict.\n"
    if not 'annotations' in current_dict[image_id].keys(): 
      current_dict[image_id]['annotations'] = [(caption, ix)]
    else:
      current_dict[image_id]['annotations'].append((caption, ix))
  return current_dict

def make_split(word_groups):
  #words is a list of tuples that cannot cooccur in the same sentence.
	#e.g. [(brown, dog), (black, cat), (brown, bear)]

  #combine train and val json files
  train_json_file = coco_anno_path %('train')  
  val_json_file = coco_anno_path %('val') 
  train_json = open(train_json_file).read()
  val_json = open(val_json_file).read()
  train_captions = json.loads(train_json) 
  val_captions = json.loads(val_json)
 
 
  #combine json dicts
  #json keys: info can be immeidately combined, licenses can be combined, type can be combined, should store images by id dict should look like
  # trainval[image_id]['image']['path']
  # trainval[image_id]['image']['list_idx'] = list index
  # trainval[image_id]['annotations']= tuple of annotations and indices
  # trainval[image_id]['json'] = train or test json
  #trainval_dict = {}
  #trainval_dict = add_json_dict(trainval_dict, train_captions, 'train_captions')
  #trainval_dict = add_json_dict(trainval_dict, val_captions, 'val_captions')
  #with open('trainval_dict.json','w') as outfile:
  #  json.dump(trainval_dict, outfile)
  trainval_json = open('trainval_dict.json').read()
  trainval_dict = json.loads(trainval_json)
  
  #determine which images have annotations which say listed words.  These images will be placed in the train set.  Other images will be split 70/30 train/test

  def add_json_dict(im, add_json):
    if trainval_dict[im]['json'] == 'train_captions':  old_dict = train_captions
    else:  old_dict = val_captions
    im_list_index = trainval_dict[im]['list_index']
    add_json['images'].append(old_dict['images'][im_list_index])
    for annotation in trainval_dict[im]['annotations']:
      add_json['annotations'].append(old_dict['annotations'][annotation[1]])
    return add_json

  def init(json_dict):
    json_dict['info'] = train_captions['info'] 
    json_dict['licenses'] = train_captions['info'] 
    json_dict['type'] = train_captions['type']
    json_dict['annotations'] = [] 
    json_dict['images'] = []
    return json_dict  

  new_train_json = {}
  new_val_json = {}
  new_val_json_newVocab = {}
  new_val_json_oldVocab = {}
  new_train_json = init(new_train_json)
  new_val_json = init(new_val_json)
  new_val_json_newVocab = init(new_val_json_newVocab)
  new_val_json_oldVocab = init(new_val_json_oldVocab) 
  other_ims = []

  for im in trainval_dict.keys():
    flag = 0
    for annotation in trainval_dict[im]['annotations']:
      a = annotation[0]
      for word_group in word_groups: 
        find_w0 = any(word_group[0] == word for word in a.lower().replace('.','').replace(',','').split(' ')) 
        find_w1 = any(word_group[1] == word for word in a.lower().replace('.','').replace(',','').split(' ')) 
        find_w = (find_w0 & find_w1)
        if find_w: flag += 1 
    if flag > 0:
      #this means one of the word groups was flagged so sample should be put in test; all other samples will be put into train then split later
      new_val_json = add_json_dict(im, new_val_json)
      new_val_json_newVocab = add_json_dict(im, new_val_json_newVocab)
    else:
      other_ims.append(im) 
     
  random.shuffle(other_ims)
    
  train_ims = other_ims[:int(0.7*len(other_ims))]
  val_ims = other_ims[int(0.7*len(other_ims)):]
  for ti in train_ims:
    new_train_json = add_json_dict(ti, new_train_json)
  for vi in val_ims:
    new_val_json = add_json_dict(vi, new_val_json) 
    new_val_json_oldVocab = add_json_dict(vi, new_val_json_oldVocab) 
  return new_train_json, new_val_json, new_val_json_newVocab, new_val_json_oldVocab

def write_txt_file(save_file, im_dir, json_dict):
  write_file = open(save_file, 'wb')
  known_ids = []
  im_dir_full = '/y/lisaanne/coco/images2/%s2014' % im_dir
  os.mkdir(im_dir_full)
  for im in json_dict['images']:
    if str(im['id']) not in known_ids:
      known_ids.append(str(im['id']))
      val_or_train = im['file_name'].split('_')[1]
      real_path = '/y/lisaanne/coco/images/%s/%s'  %(val_or_train, im['file_name'])
      link_path = '%s/%s' % (im_dir_full, im['file_name'])
      os.symlink(real_path, link_path) 
      write_file.writelines('%s\n' %str(im['id']))
  write_file.close() 

if __name__ == "__main__":
  word_groups = [('black', 'bike'), ('blue', 'train'), ('red', 'car'), ('yellow', 'shirt'), ('green', 'car')]
  train_json, val_json, val_json_newVocab, val_json_oldVocab = make_split(word_groups)
  identifier = 'fixVocab.fixFlag.'
  for w in word_groups:
    identifier += '%s_%s.' %(w[0], w[1])
  file_train_save = coco_anno_path %(identifier + 'train')
  file_val_save = coco_anno_path %(identifier + 'val')
  file_val_save_new = coco_anno_path %(identifier + 'val_novel')
  file_val_save_old = coco_anno_path %(identifier + 'val_train')
  txt_train_save = coco_txt_path %(identifier + 'train')
  txt_val_save = coco_txt_path %(identifier + 'val')
  txt_val_save_new = coco_txt_path %(identifier + 'val_novel')
  txt_val_save_old = coco_txt_path %(identifier + 'val_train')
  
  with open(file_train_save,'w') as outfile:
    json.dump(train_json, outfile)
  with open(file_val_save,'w') as outfile:
    json.dump(val_json, outfile)
  with open(file_val_save_new,'w') as outfile:
    json.dump(val_json_newVocab, outfile)
  with open(file_val_save_old,'w') as outfile:
    json.dump(val_json_oldVocab, outfile)

  #use json['images']['id'] to write text file for training with new dataset split  
  write_txt_file(txt_train_save, identifier+'train', train_json)
  write_txt_file(txt_val_save, identifier + 'val', val_json)
  write_txt_file(txt_val_save_new, identifier + 'val_novel', val_json_newVocab)  
  write_txt_file(txt_val_save_old, identifier + 'val_train', val_json_oldVocab)  


