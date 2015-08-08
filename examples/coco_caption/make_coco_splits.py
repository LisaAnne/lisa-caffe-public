import h5py
import sys
import random
import json
import os
import pickle as pkl

COCO_PATH = '../../data/coco/coco/'
feature_dir = '/y/lisaanne/image_captioning/coco_features/'
save_dir = 'h5_data/buffer_100/'
coco_anno_path = '%s/annotations/captions_%%s2014.json' % COCO_PATH
coco_txt_path = '/home/lisaanne/caffe-LSTM/data/coco/coco2014_cocoid.%s.txt'
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
  im_dir_full = '../../data/coco/coco/images/%s2014' % im_dir
  os.mkdir(im_dir_full)
  for im in json_dict['images']:
    if str(im['id']) not in known_ids:
      known_ids.append(str(im['id']))
      val_or_train = im['file_name'].split('_')[1]
      real_path = '../../data/coco/coco/images/%s/%s'  %(val_or_train, im['file_name'])
      link_path = '%s/%s' % (im_dir_full, im['file_name'])
      os.symlink(real_path, link_path) 
      write_file.writelines('%s\n' %str(im['id']))
  write_file.close() 

#add "dumb" captions to new_train_json
def augment_captions(train_dict, rm_word=None): 
  #go through each iamge
  #look at annotations
  #for each image find words that are in NN attribute list
  #add caption "A __" for each NN in the image

  nouns = pkl.load(open('../coco_attribute/attribute_lists/attributes_NN300.pkl','rb')) 
  anno_ids = [train_dict['annotations'][i]['image_id'] for i in range(len(train_dict['annotations']))] 
  id_count = 500000
  rm_ids = []
  for count_im, im in enumerate(train_dict['images'][280:290]):
    if count_im % 100 == 0:
      sys.stdout.write("\rAdding sentences for im %d/%d" % (count_im, len(train_dict['images'])))
      sys.stdout.flush()

    im_id = im['id']
    anno_idxs = [ix for ix, anno_id in enumerate(anno_ids) if anno_id == im_id]
    words = []
    for idx in anno_idxs:
      c = train_dict['annotations'][idx]['caption'].replace('.','').replace(',','').replace("'",'').lower()
      words.extend(c.split(' '))
    words = list(set(words))
    if rm_word:  #remove all annotations if one related annotation contains given word
      if rm_word in words: 
        rm_ids.extend(anno_idxs)
    word_sentences = ['A %s.' %(word) for word in words if word in nouns]
    for ws in word_sentences:
      new_annotation = {} 
      new_annotation['caption'] = ws
      new_annotation['id'] = id_count
      id_count += 1
      new_annotation['image_id'] = im_id
      train_dict['annotations'].append(new_annotation)
  if rm_word:
    for rm_id in sorted(rm_ids)[::-1]:
      a = train_dict['annotations'].pop(rm_id)
  random.shuffle(train_dict['annotations'])  
  return train_dict

def save_files(dump_json, identifier):
  file_save = coco_anno_path %(identifier)
  txt_save = coco_txt_path %(identifier)
  with open(file_save, 'w') as outfile:
    json.dump(dump_json, outfile)
  write_txt_file(txt_save, identifier, dump_json)
  print 'Wrote json to %s.\n' %file_save
  print 'Wrote image txt to %s.\n' %txt_save

if __name__ == "__main__":
  train_json_file = coco_anno_path %('train')  
  train_json = open(train_json_file).read()
  train_captions = json.loads(train_json) 

  #This will make a train set in which all 'real' zebra captions are removed
  augment_captions = augment_captions(train_captions, 'zebra')
  save_files(augment_captions, 'captions_augment_train_set_NN300_noZebra_train')

  #This will make sets where 'zebra' only occurs in val
#  word_groups = [('zebra', 'zebra')]
#  identifier = 'noZebra'
#  train_json, val_json, val_json_newVocab, val_json_oldVocab = make_split(word_groups)
#  save_files(train_json, identifier + 'train')
#  save_files(val_json, identifier + 'val')
#  save_files(val_json_newVocab, identifier + 'val_novel')
#  save_files(val_json_oldVocab, identifier + 'val_train')


#Make first compositionality split
#  word_groups = [('black', 'bike'), ('blue', 'train'), ('red', 'car'), ('yellow', 'shirt'), ('green', 'car')]
#  train_json, val_json, val_json_newVocab, val_json_oldVocab = make_split(word_groups)
#  identifier = 'fixVocab.fixFlag.'
#  for w in word_groups:
#    identifier += '%s_%s.' %(w[0], w[1])
#  file_train_save = coco_anno_path %(identifier + 'train')
#  file_val_save = coco_anno_path %(identifier + 'val')
#  file_val_save_new = coco_anno_path %(identifier + 'val_novel')
#  file_val_save_old = coco_anno_path %(identifier + 'val_train')
#  txt_train_save = coco_txt_path %(identifier + 'train')
#  txt_val_save = coco_txt_path %(identifier + 'val')
#  txt_val_save_new = coco_txt_path %(identifier + 'val_novel')
#  txt_val_save_old = coco_txt_path %(identifier + 'val_train')
#  
#  with open(file_train_save,'w') as outfile:
#    json.dump(train_json, outfile)
#  with open(file_val_save,'w') as outfile:
#    json.dump(val_json, outfile)
#  with open(file_val_save_new,'w') as outfile:
#    json.dump(val_json_newVocab, outfile)
#  with open(file_val_save_old,'w') as outfile:
#    json.dump(val_json_oldVocab, outfile)
#
#  #use json['images']['id'] to write text file for training with new dataset split  
#  write_txt_file(txt_train_save, identifier+'train', train_json)
#  write_txt_file(txt_val_save, identifier + 'val', val_json)
#  write_txt_file(txt_val_save_new, identifier + 'val_novel', val_json_newVocab)  
#  write_txt_file(txt_val_save_old, identifier + 'val_train', val_json_oldVocab)  


