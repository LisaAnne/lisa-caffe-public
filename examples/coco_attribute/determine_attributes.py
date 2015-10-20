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

def write_hdf5_file(my_dict, h5_file, rm_word=None):
  labels = np.ones((len(my_dict.keys()), len(my_dict[my_dict.keys()[0]])))*-1
  for ix, key in enumerate(my_dict.keys()):
      if ix % 50 == 0:
        sys.stdout.write("\rOn image %d/%d" %(ix, len(my_dict.keys())))
        sys.stdout.flush()
      if rm_word:
        rm_words_present = [my_dict[key][rw] for rw in rm_word]
        if any(rm_words_present) == 1:
          labels[ix, rm_word] = 1
        else: 
          labels[ix, :] = my_dict[key]
      else: 
        labels[ix, :] = my_dict[key]
  print '\n'
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

def write_image_list_2(h5_file_train, image_list_txt, json_captions, coco_folder):
  jpg_dict = {}
  for images in json_captions['images']:
    jpg_dict[images['id']] = images['file_name']
  txt_save = open(image_list_txt, 'wb')
  h5_file = h5py.File(h5_file_train, 'r')
  for image_id in h5_file['image_id']:
    txt_save.writelines('COCO_%s2014_%012d.jpg %s\n' %(coco_folder, image_id, image_id))
  txt_save.close()
  h5_file.close()  

def write_image_list_json(image_dict, image_name_template, save_json, attribute_list):
  image_list_json = {}
  for key in image_dict.keys():
    full_key = image_name_template %key
    image_list_json[full_key] = {}
    neg_labels = np.where(np.array(image_dict[key]) == 0)[0] 
    pos_labels = np.where(np.array(image_dict[key]) == 1)[0]
    image_list_json[full_key]['negative_label'] = [attribute_list[i] for i in neg_labels] 
    image_list_json[full_key]['positive_label'] = [attribute_list[i] for i in pos_labels] 
  with open(save_json, 'w') as outfile:
    json.dump(image_list_json, outfile)
  print 'Wrote file to: ', save_json

#Step 1: Sort words by type determined by classifier
#txt_file = 'train_captions_parse.out'
#read_txt = open(txt_file, 'rb')
#lines = read_txt.readlines()
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
#attributes = pkl.load(open('attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))
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

#Determine attributes
##########################################################################
#Third attempt:
#Pick all nouns, verbs, and adjectives that occur more than 200 times:
# for NN this is 511 words (puppy occurs 201 times)
# for JJ this is 155 words (checkered occurs 200 times)
# for VB still pick 100 since verbs are so infrequent (tell occurs 22 times)

#attributes = sorted_keys_dict['NN'][-511:] + sorted_keys_dict['JJ'][-155:] + sorted_keys_dict['VB'][-100:]
#attributes = list(set(attributes)) 
#pkl.dump(attributes, open('attributes_JJ155_NN511_VB100.pkl','wb')) 
attributes = pkl.load(open('attribute_lists/attributes_JJ155_NN511_VB100.pkl','rb'))
##########################################################################

#create image dict which combines image ids with labels
json_file_train = '/home/lisaanne/caffe-LSTM/data/coco/coco/annotations/captions_train2014.json'
json_file_val_val = '/home/lisaanne/caffe-LSTM/data/coco/coco/annotations/captions_val_val2014.json'
json_file_val_test = '/home/lisaanne/caffe-LSTM/data/coco/coco/annotations/captions_val_test2014.json'

json_open_train = open(json_file_train).read()
json_captions_train = json.loads(json_open_train)
json_open_val_val = open(json_file_val_val).read()
json_captions_val_val = json.loads(json_open_val_val)
json_open_val_test = open(json_file_val_test).read()
json_captions_val_test = json.loads(json_open_val_test)
image_dict_train = create_image_dict(attributes, json_captions_train, image_dict_pkl='image_dict_train_JJ155_NN511_VB100.pkl')
image_dict_val_val = create_image_dict(attributes, json_captions_val_val, image_dict_pkl='image_dict_val_val_JJ155_NN511_VB100.pkl')
#image_dict_test = create_image_dict(attributes, json_captions_test, image_dict_pkl='image_dict_%stest_JJ100_NN300_VB100.pkl' %tag)

write_image_list_train_json = '../captions_add_new_word/utils_trainAttributes/imageJson_JJ155_NN511_VB100_train.json'
write_image_list_val_val_json = '../captions_add_new_word/utils_trainAttributes/imageJson_JJ155_NN511_VB100_val_val.json'

image_name_template_train = 'train2014/COCO_train2014_%012d.jpg'
image_name_template_val_val = 'val2014/COCO_val2014_%012d.jpg'

write_image_list_json(image_dict_train, image_name_template_train, write_image_list_train_json, attributes)
write_image_list_json(image_dict_val_val, image_name_template_val_val, write_image_list_val_val_json, attributes)
