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
    return image_dict
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

def write_hdf5_file_vocab(my_dict, h5_file, attributes, vocab_file, im_list, new_word=None):
  #exact same as above but shares a vocabulary for sentnece generation

  batch_size = 10000

  #read vocab
  vocab_lines = open(vocab_file, 'rb').readlines()
  vocab_lines = [v.strip() for v in vocab_lines]
  vocab_lines = ['<EOS>'] + vocab_lines

  #read im list
  im_list_lines = open(im_list, 'rb').readlines()
  im_list_lines = [i.split(' ')[0] for i in im_list_lines]

  attribute_to_vocab_idx = [None]*len(attributes)
  for ix, attribute in enumerate(attributes):
    attribute_to_vocab_idx[ix] = vocab_lines.index(attribute)
  
  if new_word:
    new_word_idx = attributes.index(new_word)

  for i in range(0, len(im_list_lines), batch_size): 
    print 'On batch %d of %d.' %(i, len(im_list_lines)/batch_size)
    batch_end = min(i+batch_size, len(im_list_lines))
    labels = np.ones((batch_end-i, len(vocab_lines)))*-1
    for ix, im in enumerate(im_list_lines[i:batch_end]):
      im_id = int(im.split('/')[-1].split('_')[-1].split('.jpg')[0])
      if new_word:
        if my_dict[im_id][new_word_idx] == 1:
          labels[ix, vocab_lines.index(new_word)] = 1
        else: 
          labels[ix, attribute_to_vocab_idx] = my_dict[im_id]
      else: 
        labels[ix, attribute_to_vocab_idx] = my_dict[im_id]
      image_id = filter(int,my_dict.keys())
    f = h5py.File('%s_%s.h5' %(h5_file, i/batch_size))
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
#txt_file = 'train_captions_parse.out'
#read_txt = open(txt_file, 'rb')
#lines = read_txt.readlines()
#word_dict = make_word_dict(lines)
#pkl.dump(word_dict, open('word_dict.pkl','wb'))
#word_dict = pkl.load(open('attribute_lists/word_dict.pkl','rb'))
#
#attribute_pos = ['JJ', 'NN', 'VB'] #which parts of speech to keep in attribute layer
#sorted_keys_dict = {}
#for a in attribute_pos:
#  sorted_keys_dict[a] = sort_keys(word_dict[a])

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
#  first 300 nouns.  300th most used noun appears 476 times.
#  I chose this set to have some amount of evenness between how frequently words will be seen
#attributes = sorted_keys_dict['NN'][-300:]
#attributes = list(set(attributes)) 
#pkl.dump(attributes, open('attributes_NN300.pkl','wb'))
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
tag = 'basic_caption_zebra_'
image_dict_train = create_image_dict(attributes, json_captions_train, image_dict_pkl='image_dict_train_JJ100_NN300_VB100.pkl')
#image_dict_val = create_image_dict(attributes, json_captions_val, image_dict_pkl='image_dict_%sval_JJ100_NN300_VB100.pkl' %tag)
#image_dict_test = create_image_dict(attributes, json_captions_test, image_dict_pkl='image_dict_%stest_JJ100_NN300_VB100.pkl' %tag)

#save h5 files and txt files
h5_file_train = '/x/lisaanne/coco_attribute/utils_trainAttributes/attributes_vocab8800_JJ100_NN300_VB100_train_basic_caption'
h5_file_val = 'utils_trainAttributes/attributes_vocab8800_JJ100_NN300_VB100_val'
h5_file_test = 'utils_trainAttributes/attributes_vocab8800_JJ100_NN300_VB100_test'
image_list_txt_train = 'utils_trainAttributes/attributes_vocab8800_JJ100_NN300_VB100_imageList_train.txt'
image_list_txt_val = 'utils_trainAttributes/attributes_vocab8800_JJ100_NN300_VB100_imageList_val.txt'
image_list_txt_test = 'utils_trainAttributes/attributes_vocab8800_JJ100_NN300_VB100_imageList_test.txt'
vocab_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'
image_list_base = '../coco_caption/h5_data/buffer_100/%s%s_aligned_20_batches/image_list.with_dummy_labels.txt'

write_hdf5_file_vocab(image_dict_train, h5_file_train, attributes, vocab_file, image_list_base %(tag, 'train'), new_word='zebra')
print 'Wrote hdf5 file to %s.\n' %h5_file_train
#write_hdf5_file_vocab(image_dict_val, h5_file_val, attributes, vocab_file, image_list_base %(tag, 'val'))
#print 'Wrote hdf5 file to %s.\n' %h5_file_val
#write_hdf5_file_vocab(image_dict_test, h5_file_test)
#print 'Wrote hdf5 file to %s.\n' %h5_file_test
#write_image_list(image_dict_train, image_list_txt_train, json_captions_train)
#print 'Wrote image list train text to %s.\n' %image_list_txt_train
#write_image_list(image_dict_val, image_list_txt_val, json_captions_val)
#print 'Wrote image list train text to %s.\n' %image_list_txt_val
#write_image_list(image_dict_test, image_list_txt_test, json_captions_test)
#print 'Wrote image list test text to %s.\n' %image_list_txt_test
