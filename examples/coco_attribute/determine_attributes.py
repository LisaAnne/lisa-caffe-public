import sys
import copy
import pickle as pkl
import json
import numpy as np
import h5py

def make_word_dict(lines):
  word_dict = {}
  for ix, line in enumerate(lines):
    if ix % 10000 == 0:
      sys.stdout.write('\rOn line %d/%d.' %(ix, len(lines)))
      sys.stdout.flush()
    line = line.split(' ')
    line = line[:-2]  #get rid of ._.
    for word in line: 
      pos_pair = word.split('_')
      w = pos_pair[0]
      pos = pos_pair[1]
      if pos in word_dict.keys():
        if w in word_dict[pos].keys():
          word_dict[pos][w] += 1
        else: 
          word_dict[pos][w] = 0  
      else:
        word_dict[pos] = {}
        word_dict[pos][w] = 1
  sys.stdout.write('\n')
  return word_dict 

def sort_keys(word_dict):
  #sorts words in a dict by how frequently they appear
  cmp_count = lambda x,y: cmp(word_dict[x], word_dict[y])
  return sorted(word_dict, cmp_count)

def create_image_dict(attributes, json_captions, image_dict_pkl=None, save_name=None):
  if image_dict_pkl:
    image_dict = pkl.load(open(image_dict_pkl,'rb'))
  else:
    image_dict = {}
    for ix, caption in enumerate(json_captions['annotations']):
      if ix % 100 == 0:
        sys.stdout.write('\rOn caption %d/%d.' %(ix, len(json_captions['annotations'])))
        sys.stdout.flush()
      c = caption['caption']
      c = c.replace('.','').replace(',','').replace("'",'').lower()
      image_id = caption['image_id'] 
      attribute_labels = [any([a2 == c2 for c2 in c.split()]) for a2 in attributes] 
      if image_id in image_dict.keys():
        image_dict[image_id] = [image_dict[image_id][i] + attribute_labels[i] for i in range(len(attribute_labels))]
      else:
        image_dict[image_id] = attribute_labels
  for key in image_dict.keys():
    image_dict[key] = [min(x,1) for x in image_dict[key]]
  if save_name:
    pkl.dump(image_dict, open(save_name, 'wb'))
  return image_dict

def write_hdf5_file(my_dict, h5_file):
  labels = np.zeros((len(my_dict.keys()), len(my_dict[my_dict.keys()[0]])))
  for ix, key in enumerate(my_dict.keys()):
    labels[ix,:] = my_dict[key]
  image_id = filter(int,my_dict.keys())
  f = h5py.File(h5_file)
  f.create_dataset('labels', data=labels)
  f.create_dataset('image_id', data=image_id)
  f.close()

def write_image_list(my_dict, image_list_txt, json_captions):
  jpg_dict = {}
  for images in json_captions['images']:
    jpg_dict[images['id']] = images['file_name']
  txt_save = open(image_list_txt, 'wb')
  for key in my_dict.keys():
    txt_save.writelines('%s %s\n' %(jpg_dict[key], key))
  txt_save.close()
 
#Step 1: Sort words by type determined by classifier
txt_file = 'train_captions_parse.out'
read_txt = open(txt_file, 'rb')
lines = read_txt.readlines()
#word_dict = make_word_dict(lines)
#pkl.dump(word_dict, open('word_dict.pkl','wb'))
word_dict = pkl.load(open('attribute_lists/word_dict.pkl','rb'))

attribute_pos = ['JJ', 'NN', 'VB'] #which parts of speech to keep in attribute layer
sorted_keys_dict = {}
for a in attribute_pos:
  sorted_keys_dict[a] = sort_keys(word_dict[a])

#Determine attributes
##########################################################################
#First attempt:
#somewhat arbitrarily picked the following words:
#  first 100 verbs.  100th most used verb 'tell' only appears 22 times in the gt sentences
#  first 100 adj.  100th most used adj 'cluttered' appears 381 times
#  first 300 nouns.  300th most used noun appears 476 times.
#  I chose this set to have some amount of evenness between how frequently words will be seen
#attributes = sorted_keys_dict['JJ'][-100:] + sorted_keys_dict['NN'][-300:] + sorted_keys_dict['VB'][-100:]
#attributes = list(set(attributes))
#pkl.dump(attributes, open('attributes_JJ100_NN300_VB100.pkl','wb'))
attributes = pkl.load(open('attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))
##########################################################################

#Determine attributes
##########################################################################
#Second attempt:
#somewhat arbitrarily picked the following words:
#  first 100 verbs.  100th most used verb 'tell' only appears 22 times in the gt sentences
#  first 100 adj.  100th most used adj 'cluttered' appears 381 times
#  first 300 nouns.  300th most used noun appears 476 times.
#  I chose this set to have some amount of evenness between how frequently words will be seen
attributes = sorted_keys_dict['NN'][-300:]
attributes = list(set(attributes))
pkl.dump(attributes, open('attributes_NN300.pkl','wb'))
#attributes = pkl.load(open('attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))
##########################################################################

#create image dict which combines image ids with labels
json_file_train = '/home/lisaanne/caffe-LSTM/data/coco/coco/annotations/captions_train2014.json'
json_file_val = '/home/lisaanne/caffe-LSTM/data/coco/coco/annotations/captions_val2014.json'
json_file_test = '/home/lisaanne/caffe-LSTM/data/coco/coco/annotations/captions_test2014.json'
json_open_train = open(json_file_train).read()
json_captions_train = json.loads(json_open_train)
json_open_val = open(json_file_val).read()
json_captions_val = json.loads(json_open_val)
json_open_test = open(json_file_test).read()
json_captions_test = json.loads(json_open_test)
image_dict_train = create_image_dict(attributes, json_captions_train, save_name='image_dict_train_NN300.pkl')
image_dict_val = create_image_dict(attributes, json_captions_val, save_name='image_dict_val_NN300.pkl')
image_dict_test = create_image_dict(attributes, json_captions_test, save_name='image_dict_test_NN300.pkl')

#save h5 files and txt files
h5_file_train = 'utils_trainAttributes/attributes_NN300_train.h5'
h5_file_val = 'utils_trainAttributes/attributes_NN300_val.h5'
h5_file_test = 'utils_trainAttributes/attributes_NN300_test.h5'
image_list_txt_train = 'utils_trainAttributes/attributes_NN300_imageList_train.txt'
image_list_txt_val = 'utils_trainAttributes/attributes_NN300_imageList_val.txt'
image_list_txt_test = 'utils_trainAttributes/attributes_NN300_imageList_test.txt'

write_hdf5_file(image_dict_train, h5_file_train)
print 'Wrote hdf5 file to %s.\n' %h5_file_train
write_hdf5_file(image_dict_val, h5_file_val)
print 'Wrote hdf5 file to %s.\n' %h5_file_val
write_hdf5_file(image_dict_test, h5_file_test)
print 'Wrote hdf5 file to %s.\n' %h5_file_test
write_image_list(image_dict_train, image_list_txt_train, json_captions_train)
print 'Wrote image list train text to %s.\n' %image_list_txt_train
write_image_list(image_dict_val, image_list_txt_val, json_captions_val)
print 'Wrote image list train text to %s.\n' %image_list_txt_val
write_image_list(image_dict_test, image_list_txt_test, json_captions_test)
print 'Wrote image list test text to %s.\n' %image_list_txt_test
