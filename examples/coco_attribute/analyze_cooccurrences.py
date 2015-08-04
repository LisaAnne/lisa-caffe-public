import sys
import copy
import pickle as pkl
import json
import numpy as np
import h5py

def make_cooccurrence_matrix(captions):
  vocabulary_file = open('../coco_caption/h5_data/buffer_100/vocabulary.txt','rb')
  vocabulary = vocabulary_file.readlines()
  num_words = len(vocabulary)
  vocab_dict = {}
  for iw, word in enumerate(vocabulary):
    vocab_dict[word.replace('\n','')] = iw
  word_matrix = np.zeros((num_words, num_words))
  for ix, caption in enumerate(captions):
    if ix % 1000 == 0:
      sys.stdout.write('\rOn caption %d/%d.' %(ix, len(captions)))
      sys.stdout.flush()
    c = caption['caption'] 
    c = c.replace('.','').replace(',','').replace("'",'').lower().split(' ')
    illegal_words = ['']
    for i_w1, w1 in enumerate(c):
      for i_w2, w2 in enumerate(c):
        if not i_w1 == i_w2:
          if not ((w1 in illegal_words) or (w2 in illegal_words)): 
            word_matrix[vocab_dict[w1],vocab_dict[w2]] += 1
            word_matrix[vocab_dict[w2], vocab_dict[w1]] += 1
  sys.stdout.write('\n')
  return word_matrix, vocab_dict 

#Analyze cooccurrences that occur in generated val set

json_file_val = 'generation_result.json'
json_open_val = open(json_file_val).read()
json_captions_val = json.loads(json_open_val)

#Come up with cooccurrence matrix and dictionary
word_matrix, vocab_dict = make_cooccurrence_matrix(json_captions_val)
cooccurrences = {}
cooccurrences['word_matrix'] = word_matrix
cooccurrences['vocab_dict'] = vocab_dict
pkl.dump(cooccurrences, open('cooccurrence_matrix_val.p','wb'))

