import sys
sys.path.insert(0,'../../python')
sys.path.insert(0, '../word_similarity/')
import caffe
from init import *
import numpy as np
import pickle as pkl
from w2vDist import *

save_tag = 'da'
adapt_embed = False 
num_close_words_im = 1
num_close_words_lm = 1

eightyK = False

#zebra
add_words = {'words': ['zebra', 'zebras'], 'classifiers': ['zebra', 'zebra'], 'illegal_words': ['zebra']}

model = 'mrnn_attributes_fc8.direct.from_features.wtd.prototxt' #The wtd has all the parameters we care about and will take up less memory
if not eightyK: 
  model_weights_lm = pretrained_lm + 'mrnn.direct_iter_110000'
else:
  model_weights_lm = pretrained_lm + 'mrnn.lm.direct_imtextyt_lr0.01_iter_120000'
  #model_weights_lm = pretrained_lm + 'mrnn.lm.direct_surf_lr0.01_iter_120000'
  #model_weights_lm = pretrained_lm + 'mrnn.lm.direct_wikisent_lr0.01_iter_120000'

#model_weights_trained = '/z/lisaanne/snapshots_caption_models/attributes_JJ100_NN300_VB100_zebra_cocoImages_captions_ftLMPretrain_iter_110000'
#model_weights_trained = '/z/lisaanne/snapshots_caption_models/attributes_JJ100_NN300_VB100_zebra_cocoImages_captions_ftLMPretrain_pretrainLM50_iter_110000'
model_weights_trained = '/z/lisaanne/snapshots_caption_models/attributes_JJ100_NN300_VB100_zebra_cocoImages_captions_ftLMPretrain_pretrainLM50_lr0p001_iter_10000'

net_lm = caffe.Net(model, model_weights_lm + '.caffemodel', caffe.TEST)
net_trained = caffe.Net(model, model_weights_trained + '.caffemodel', caffe.TEST)

if not eightyK:
  vocab = open('../coco_caption/h5_data/buffer_100/vocabulary.txt').readlines()
else:
  vocab = open('/x/lisaanne/pretrained_lm/yt_coco_surface_80k_vocab.txt').readlines()
vocab = [v.strip() for v in vocab]
vocab = ['EOS'] + vocab

im_bias = False
if len(net_trained.params['predict-im']) > 1:
  im_bias = True

attributes = pkl.load(open('../coco_attribute/attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))

#create word to vec
W2V = w2v()
W2V.readVectors()
W2V.reduce_vectors(attributes)

def closeness_embedding(new_word):
  return W2V.findClosestWords(new_word)

close_words_im = {}
close_words_lm = {}

for aw, word in enumerate(add_words['words']):
  close_words_im[word] = {}
  word_sims = closeness_embedding(add_words['classifiers'][aw])
  for illegal_word in add_words['illegal_words']:
    illegal_idx = attributes.index(illegal_word)
    word_sims[illegal_idx] = -100000

  close_words_im[word] = {}
  close_words_lm[word] = {}

  close_words_im[word]['close_words'] = [attributes[i] for i in np.argsort(word_sims)[-num_close_words_im:]]
  close_words_lm[word]['close_words'] = [attributes[i] for i in np.argsort(word_sims)[-num_close_words_lm:]]
  close_words_im[word]['weights'] = [1./num_close_words_im]*num_close_words_im
  close_words_lm[word]['weights'] = [1./num_close_words_lm]*num_close_words_im

for add_word, add_word_classifier in zip(add_words['words'], add_words['classifiers']):
  #transfer im
  #collect delta on lm

  transfer_word_lm = close_words_lm[add_word]
  transfer_word_im = close_words_im[add_word]

  add_word_idx = vocab.index(add_word)
  delta_weights = np.zeros(net_trained.params['predict'][0].data.shape[1],)
  delta_bias = 0
  for twl in transfer_word_lm['close_words']:
    transfer_word_lm_idx = vocab.index(twl)
    delta_transfer_weights = net_trained.params['predict'][0].data[transfer_word_lm_idx,:] - net_lm.params['predict'][0].data[transfer_word_lm_idx,:]
    delta_transfer_bias = net_trained.params['predict'][1].data[transfer_word_lm_idx] - net_lm.params['predict'][1].data[transfer_word_lm_idx]
    delta_weights += delta_transfer_weights
    delta_bias += delta_transfer_bias

  net_trained.params['predict'][0].data[add_word_idx,:] += delta_transfer_weights/num_close_words_lm
  net_trained.params['predict'][1].data[add_word_idx] += delta_transfer_bias/num_close_words_lm
  
  #transfer im (same as before -- just do direct transfer)

  weights_im_transfer = np.zeros(net_trained.params['predict-im'][0].data.shape[1],)
  if im_bias:
    bias_im_transfer = 0
 
  for twi in transfer_word_im['close_words']:
    transfer_word_im_idx = vocab.index(twi)
    weights_im_transfer += net_trained.params['predict-im'][0].data[transfer_word_im_idx,:]
    if im_bias:
      bias_im_transfer += net_trained.params['predict-im'][1].data[transfer_word_im_idx]
    
  net_trained.params['predict-im'][0].data[add_word_idx,:] = weights_im_transfer/num_close_words_im
  if im_bias:
    net_trained.params['predict-im'][1].data[add_word_idx] = bias_im_transfer/num_close_words_im

  add_attribute_idx = attributes.index(add_word_classifier)  
  self_correlation = 0
  for twi in transfer_word_im['close_words']:
    transfer_attribute_idx = attributes.index(twi)  
    #co-correlation 
    self_correlation += net_trained.params['predict-im'][0].data[transfer_word_im_idx, transfer_attribute_idx]
    net_trained.params['predict-im'][0].data[add_word_idx, transfer_attribute_idx] = 0 
    net_trained.params['predict-im'][0].data[transfer_word_im_idx, add_attribute_idx] = 0 

  net_trained.params['predict-im'][0].data[add_word_idx, add_attribute_idx] = self_correlation/num_close_words_im 
  

net_trained.save('%s.%s.caffemodel' %(model_weights_trained, save_tag))
print 'Saved at: %s.%s.caffemodel' %(model_weights_trained, save_tag)
  

