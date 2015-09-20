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

def init(json_dict):
  json_dict['info'] = train_captions['info'] 
  json_dict['licenses'] = train_captions['info'] 
  json_dict['type'] = train_captions['type']
  json_dict['annotations'] = [] 
  json_dict['images'] = []
  return json_dict  

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

def write_txt_file(save_file, im_dir, json_dict, new_im_folder=False):
  write_file = open(save_file, 'wb')
  known_ids = []
  im_dir_full = '../../data/coco/coco/images/%s2014' % im_dir
  os.mkdir(im_dir_full)
  for im in json_dict['images']:
    if str(im['id']) not in known_ids:
      known_ids.append(str(im['id']))
      val_or_train = im['file_name'].split('_')[1]
      if new_im_folder:
        real_path = '/home/lisaanne/caffe-LSTM/data/coco/coco/images/%s/%s'  %(val_or_train, im['file_name'])
        link_path = '%s/%s' % (im_dir_full, im['file_name'])
        os.symlink(real_path, link_path) 
      write_file.writelines('%s\n' %str(im['id']))
  write_file.close() 

def match_words(rm_words, words):
  if not rm_words:
    return False #if rm_words is none want to return False.
  list_matches = [False]*len(rm_words)
  for x, rm_w in enumerate(rm_words):
    for w in words:
      if w == rm_w:
        list_matches[x] = True
  return any(list_matches)

#add "dumb" captions to new_train_json
def augment_captions(train_dict, rm_word=None, rm_all_object_sents=False, all_object_sents=False, no_annotations=False): 
  #go through each iamge
  #look at annotations
  #for each image find words that are in NN attribute list
  #add caption "A __" for each NN in the image
  #rm_word can be rm_words -- multiple words
  #rm_all_object_sents: If this is True, then for sentences with rm_word, then augmented captions will include captions for all nouns in sentence ('A zebra in the field') --> 'A zebra.' 'A field.'
  #all_object_sents: If this is true, then all training sentences are augmented with noun-captions.
  #no_annotations: remove annotations that include and of the rm_word

  nouns = pkl.load(open('../coco_attribute/attribute_lists/attributes_NN300.pkl','rb')) 
  anno_ids = [train_dict['annotations'][i]['image_id'] for i in range(len(train_dict['annotations']))] 
  id_count = 500000
  rm_ids = []
  rm_image_ids = []
  for count_im, im in enumerate(train_dict['images']):
    if count_im % 50 == 0:
      sys.stdout.write("\rAdding sentences for im %d/%d" % (count_im, len(train_dict['images'])))
      sys.stdout.flush()

    im_id = im['id']
    anno_idxs = [ix for ix, anno_id in enumerate(anno_ids) if anno_id == im_id]
    words = []
    for idx in anno_idxs:
      c = train_dict['annotations'][idx]['caption'].replace('.','').replace(',','').replace("'",'').lower()
      words.extend(c.split(' '))
    words = list(set(words))
    mw = match_words(rm_word, words)
    if mw: #If match word (match_word will return false if rm_word is None) 
      rm_ids.extend(anno_idxs)
      #Right now there are multiple sentences with objects in the image that are *not* zebra; this is not quite right
      if no_annotations:
        rm_image_ids.append(count_im)
      if not rm_all_object_sents:  #This will make the only sentences associated with an image the label
        #option2 of rm_words...
        words = rm_word 

    if (mw | all_object_sents) & (not no_annotations): #if match words or if inserting augmented sentences for all train sentences and annotations
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
  if no_annotations:
    for rm_image_id in sorted(rm_image_ids)[::-1]:
      a = train_dict['images'].pop(rm_image_id)
  random.shuffle(train_dict['annotations'])  
  return train_dict

def separate_val_set(val_dict, rm_word=None):
  val_dict_train = {}
  val_dict_novel = {}
  val_dict_train = init(val_dict_train)
  val_dict_novel = init(val_dict_novel)

  anno_ids = [val_dict['annotations'][i]['image_id'] for i in range(len(val_dict['annotations']))]
  novel_annotation_ids = [] 
  novel_image_ids = [] 
  train_annotation_ids = [] 
  train_image_ids = [] 
  for count_im, im in enumerate(val_dict['images']):
    if count_im % 50 == 0:
      sys.stdout.write("\rAdding sentences for im %d/%d" % (count_im, len(val_dict['images'])))
      sys.stdout.flush()

    im_id = im['id']
    anno_idxs = [ix for ix, anno_id in enumerate(anno_ids) if anno_id == im_id]
    words = []
    for idx in anno_idxs:
      c = val_dict['annotations'][idx]['caption'].replace('.','').replace(',','').replace("'",'').lower()
      words.extend(c.split(' '))
    words = list(set(words))
    if match_words(rm_word, words): 
      novel_annotation_ids.extend(anno_idxs)
      novel_image_ids.append(count_im)
    else:
      train_annotation_ids.extend(anno_idxs)
      train_image_ids.append(count_im)
  for i in novel_annotation_ids:
    val_dict_novel['annotations'].append(val_dict['annotations'][i])
  for i in train_annotation_ids:
    val_dict_train['annotations'].append(val_dict['annotations'][i])
  for i in novel_image_ids:
    val_dict_novel['images'].append(val_dict['images'][i])
  for i in train_image_ids:
    val_dict_train['images'].append(val_dict['images'][i])
  print '\n'
  return val_dict_novel, val_dict_train

def vocab_pretrain(train_dict):
  random.shuffle(train_dict['images'])  
  train_dict['images'] = train_dict['images'][0:int(len(train_dict['images'])*0.75)]
  train_im_ids = [t['id'] for t in train_dict['images']]
  rm_c_ids = []
  for cix, c in enumerate(train_dict['annotations']):
    if cix % 50 == 0:
      sys.stdout.write("\rAdding sentences for im %d/%d" % (cix, len(train_dict['annotations'])))
      sys.stdout.flush()
    if c['image_id'] not in train_im_ids:
      rm_c_ids.append(cix)
  print '\n'
  for rm_c_id in sorted(rm_c_ids)[::-1]:
    a = train_dict['annotations'].pop(rm_c_id)
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

  val_json_file = coco_anno_path %('val')  
  val_json = open(val_json_file).read()
  val_captions = json.loads(val_json) 
  
  test_json_file = coco_anno_path %('test')  
  test_json = open(test_json_file).read()
  test_captions = json.loads(test_json) 

  #This will make a train set in which all 'real' zebra captions are removed
  #sentences with A noun.
  tag = 'no_caption_motorcycle_'

  #rm_words = ['zebra', 'zebras']
  rm_words = ['motor', 'cycle', 'motorcycle', 'motors', 'cycles', 'motorcycles']

  #baseline
  #rm_words = None

  #augment_captions:
	# ex sentence: 'A zebra in the field', 'A cow in the grass'
        #rm word: zebra
        #rm_all_object_sents (True) --> 'A zebra.'  'A field.' 'A cow in the grass.'
        #all_object_sents (True) --> 'A zebra.' 'A field.' 'A cow.' 'A grass.' 'A cow in the grass.'
        #no_annotations (True) --> 'A cow in the grass.'
  #basic captions
  #augment_captions = augment_captions(train_captions, rm_words, rm_all_object_sents=False, all_object_sents=False, no_annotations=False)
  #no captions
  augment_captions = augment_captions(train_captions, rm_words, rm_all_object_sents=False, all_object_sents=False, no_annotations=True)

  #make smaller train set for training vocab
  #vocab_pretrain = vocab_pretrain(train_captions) 
 
  save_files(augment_captions, tag + '_train')

#  val_captions_novel, val_captions_train = separate_val_set(val_captions, rm_words)
#  save_files(val_captions_novel, tag+'val_novel')
#  save_files(val_captions_train, tag+'val_train')

#################################################################################################################


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


