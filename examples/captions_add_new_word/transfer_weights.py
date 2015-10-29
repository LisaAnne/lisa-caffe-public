import sys
sys.path.insert(0,'../../python/')
sys.path.insert(0, '../word_similarity/')
from w2vDist import *
import caffe
import numpy as np
import copy
import pickle as pkl
import hickle as hkl
from nltk.corpus import wordnet as wn
#import find_close_words

save_tag = 'closest_W2V'
eightK = False
transfer_embed = False 
num_close_words_im = 1
num_close_words_lm = 1

all_add_words = []

#zebra
#add_words = {'words': ['zebra', 'zebras'], 'classifiers': ['zebra', 'zebra'], 'illegal_words': ['zebra']}
#all_add_words.append(add_words)
#add_words = {'words': ['pizza', 'pizzas'], 'classifiers': ['pizza', 'pizza'], 'illegal_words': ['pizza']}
#all_add_words.append(add_words)
#add_words = {'words': ['suitcase', 'suitcases', 'luggage', 'luggages'], 'classifiers': ['suitcase', 'suitcase', 'luggage', 'luggage'], 'illegal_words': ['luggage', 'suitcase']}
#all_add_words.append(add_words)
#add_words = {'words': ['bottle', 'bottles'], 'classifiers': ['bottle', 'bottle'], 'illegal_words': ['bottle']}
#all_add_words.append(add_words)
add_words = {'words': ['bus', 'busses'], 'classifiers': ['bus', 'bus'], 'illegal_words': ['bus']}
all_add_words.append(add_words)
#add_words = {'words': ['couch', 'couches'], 'classifiers': ['couch', 'couch'], 'illegal_words': ['couch']}
#all_add_words.append(add_words)
#add_words = {'words': ['microwave', 'microwaves'], 'classifiers': ['microwave', 'microwave'], 'illegal_words': ['microwave']}
#all_add_words.append(add_words)
#add_words = {'words': ['racket', 'rackets', 'racquet', 'racquets'], 'classifiers': ['racket', 'racket', 'racquet', 'racquet'], 'illegal_words': ['racket', 'racquet']}
#all_add_words.append(add_words)


#Relearn image and language model
model_weights = '/z/lisaanne/snapshots_caption_models/attributes_JJ100_NN300_VB100_zebra_cocoImages_captions_noLMPretrain_dropout_iter_110000'
model='mrnn_attributes_fc8.direct.from_features.wtd.prototxt'
net = caffe.Net(model, model_weights + '.caffemodel', caffe.TRAIN)


if 'predict-lm' in net.params.keys():
  predict_lm = 'predict-lm'
else:
  predict_lm = 'predict'

if len(net.params['predict-im']) > 1:
  im_bias = True
else: 
  im_bias = False

attributes = pkl.load(open('../coco_attribute/attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))

scale_feats = False
if scale_feats:
  scale_feats = hkl.load('utils_trainAttributes/average_weights_zebra.hkl')

#LISA: This should be neater, but it will work for now
#create word to vec
W2V = w2v()
W2V.readVectors()
W2V.reduce_vectors(attributes, '-n')

#create sysnset stuff
lexical_synsets = [None]*len(attributes)
for ix, attribute in enumerate(attributes):
  if len(wn.synsets(attribute)) > 0:
    lexical_synsets[ix] = filter(lambda x: 'n' == x.pos(), wn.synsets(attribute))  

def closeness_embedding(new_word):
  return W2V.findClosestWords(new_word)

def closeness_embedding_force(new_word):
  word_sims = np.ones((471,))*-1
  force_idx = attributes.index('horse')
  word_sims[force_idx] = 100
  return W2V.findClosestWords(new_word)

def closeness_embedding_synset(new_word):
  new_word_synset = filter(lambda x: 'n' == x.pos(), wn.synsets(new_word))
  closeness = []
  for ls in lexical_synsets:
    sim_nws = 0
    for nws in new_word_synset:
      if ls:
        sim = 0
        for l in ls:
          sim += nws.path_similarity(l)
        sim_nws += sim
        div = len(ls)
      else:
        sim_nws = -10000
        div = 1
    closeness.append(sim_nws/div)
  return closeness

closeness_metric = closeness_embedding

for add_words in all_add_words:
  close_words_im = {}
  close_words_lm = {}
  model_weights = '/y/lisaanne/mrnn_direct/snapshots/attributes_JJ100_NN300_VB100_eightClusters_cocoImages_captions_fixLMPretrain_fixSplit5_iter_110000'
  model='mrnn_attributes_fc8.direct.from_features.wtd.prototxt'
  net = caffe.Net(model, model_weights + '.caffemodel', caffe.TRAIN)
  save_tag = save_tag_template % add_words['words'][0]
  for aw, word in enumerate(add_words['words']):
    close_words_im[word] = {}
    word_sims = closeness_metric(add_words['classifiers'][aw])
    for illegal_word in add_words['illegal_words']:
      illegal_idx = attributes.index(illegal_word)
      word_sims[illegal_idx] = -100000
  
    close_words_im[word] = {}
    close_words_lm[word] = {}
  
    close_words_im[word]['close_words'] = [attributes[i] for i in np.argsort(word_sims)[-num_close_words_im:]]
    close_words_lm[word]['close_words'] = [attributes[i] for i in np.argsort(word_sims)[-num_close_words_lm:]]
    close_words_im[word]['weights'] = [1./num_close_words_im]*num_close_words_im
    close_words_lm[word]['weights'] = [1./num_close_words_lm]*num_close_words_im
  
 
  vocab_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'
  vocab_lines = open(vocab_file, 'rb').readlines()
  vocab_lines = [v.strip() for v in vocab_lines]
  vocab_lines = ['<EOS>'] + vocab_lines
  
  predict_weights_lm = copy.deepcopy(net.params['predict-lm'][0].data)
  predict_bias_lm = copy.deepcopy(net.params['predict-lm'][1].data)
  predict_weights_im = copy.deepcopy(net.params['predict-im'][0].data)
  #predict_bias_im = copy.deepcopy(net.params['predict-im'][1].data)
  
  for aw, add_word in enumerate(add_words['words']):
    add_word_idx = vocab_lines.index(add_word)
    attribute_loc = attributes.index(add_words['classifiers'][aw])
    transfer_weights_lm = np.ones((predict_weights_lm.shape[1],))*0
    transfer_bias_lm = 0
    transfer_weights_im = np.ones((predict_weights_im.shape[1],))*0
  
    for wi, close_word in enumerate(close_words_im[add_word]['close_words']):
      close_word_idx = vocab_lines.index(close_word)
      transfer_weights_im += predict_weights_im[close_word_idx,:]*close_words_im[add_word]['weights'][wi]
    
    transfer_weights_im /= num_close_words_im
    
    scale = 1.
    if scale_feats:
      scale = scale_feats[close_word]/scale_feats[add_words['classifiers'][0]]
    
    #Take care of classifier cross terms
    for wi, close_word in enumerate(close_words_im[add_word]['close_words']): 
      close_word_idx = vocab_lines.index(close_word)
      close_word_attribute_loc = attributes.index(close_word)
      transfer_weights_im[attribute_loc] = predict_weights_im[close_word_idx, close_word_attribute_loc]*scale
      transfer_weights_im[close_word_attribute_loc] = 0
  
    for wi, close_word in enumerate(close_words_lm[add_word]['close_words']):
      close_word_idx = vocab_lines.index(close_word)
      transfer_weights_lm += predict_weights_lm[close_word_idx,:]*close_words_lm[add_word]['weights'][wi]
      transfer_bias_lm += predict_bias_lm[close_word_idx]*close_words_lm[add_word]['weights'][wi]
  
    transfer_weights_lm /= num_close_words_lm
    transfer_bias_lm /= num_close_words_lm
 
 
    predict_weights_lm[add_word_idx,:] = transfer_weights_lm
    predict_bias_lm[add_word_idx] = transfer_bias_lm
    predict_weights_im[add_word_idx,:] = transfer_weights_im
  
    for wi, close_word in enumerate(close_words_lm[add_word]['close_words']):
      close_word_idx = vocab_lines.index(close_word)
      predict_weights_im[close_word_idx,attribute_loc] = 0 
     
  net.params['predict-lm'][0].data[...] = predict_weights_lm
  net.params['predict-lm'][1].data[...] = predict_bias_lm
  net.params['predict-im'][0].data[...] = predict_weights_im
  #net.params['predict-im'][1].data[...] = predict_bias_im
  net.save('%s.%s.caffemodel' %(model_weights, save_tag))
  
  print close_words_im
  print close_words_lm
  print 'Saved to: %s.%s.caffemodel' %(model_weights, save_tag) 
  print 'Done.'
