import sys
sys.path.insert(0,'../../python/')
sys.path.insert(0, '../word_similarity/')
from w2vDist import *
import caffe
import numpy as np
import copy
import pickle as pkl
#import find_close_words

save_tag = 'closest_W2V_t'
transfer_embed = False
num_close_words_im = 1
num_close_words_lm = 1

#zebra
add_words = {'words': ['zebra', 'zebras'], 'classifiers': ['zebra', 'zebra'], 'illegal_words': ['zebra']}
#add_words = {'words': ['pizza', 'pizzas'], 'classifiers': ['pizza', 'pizza'], 'illegal_words': ['pizza']}
#add_words = {'words': ['luggage', 'luggages'], 'classifiers': ['luggage', 'luggage'], 'illegal_words': ['luggage', 'suitcase']}

#Relearn image and language model
model_weights = '/z/lisaanne/snapshots_caption_models/attributes_JJ100_NN300_VB100_zebra_cocoImages_captions_fixLMPretrain_iter_110000'
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
 
vocab_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'
vocab_lines = open(vocab_file, 'rb').readlines()
vocab_lines = [v.strip() for v in vocab_lines]
vocab_lines = ['<EOS>'] + vocab_lines

predict_weights_lm = copy.deepcopy(net.params[predict_lm][0].data)
predict_bias_lm = copy.deepcopy(net.params[predict_lm][1].data)
predict_weights_im = copy.deepcopy(net.params['predict-im'][0].data)
if transfer_embed:
  embed_weights = copy.deepcopy(net.params['embedding'][0].data)
if im_bias:
  predict_bias_im = copy.deepcopy(net.params['predict-im'][1].data)

for aw, add_word in enumerate(add_words['words']):
  add_word_idx = vocab_lines.index(add_word)
  attribute_loc = attributes.index(add_words['classifiers'][aw])
  transfer_weights_lm = np.ones((predict_weights_lm.shape[1],))*0
  transfer_bias_lm = 0
  transfer_weights_im = np.ones((predict_weights_im.shape[1],))*0
  if im_bias:
    transfer_bias_im = 0
  if transfer_embed: 
    embed = np.ones((embed_weights.shape[1],))*0

  for wi, close_word in enumerate(close_words_im[add_word]['close_words']):
    close_word_idx = vocab_lines.index(close_word)
    transfer_weights_im += predict_weights_im[close_word_idx,:]*close_words_im[add_word]['weights'][wi]
    if im_bias:
      transfer_bias_im += predict_bias_im[close_word_idx]*close_words_im[add_word]['weights'][wi]
  
  transfer_weights_im /= num_close_words_im
  if im_bias: 
    transfer_bias_im /= num_close_words_im
  
  #Take care of classifier cross terms
  for wi, close_word in enumerate(close_words_im[add_word]['close_words']): 
    close_word_idx = vocab_lines.index(close_word)
    close_word_attribute_loc = attributes.index(close_word)
    transfer_weights_im[attribute_loc] = predict_weights_im[close_word_idx, close_word_attribute_loc]
    transfer_weights_im[close_word_attribute_loc] = 0

  for wi, close_word in enumerate(close_words_lm[add_word]['close_words']):
    close_word_idx = vocab_lines.index(close_word)
    transfer_weights_lm += predict_weights_lm[close_word_idx,:]*close_words_lm[add_word]['weights'][wi]
    transfer_bias_lm += predict_bias_lm[close_word_idx]*close_words_lm[add_word]['weights'][wi]
    if transfer_embed:
      embed += embed_weights[close_word_idx,:]*close_words_lm[add_word]['weights']

  transfer_weights_lm /= num_close_words_lm
  transfer_bias_lm /= num_close_words_lm
  if transfer_embed:
    embed /= num_close_words_lm

  predict_weights_lm[add_word_idx,:] = transfer_weights_lm
  predict_bias_lm[add_word_idx] = transfer_bias_lm
  predict_weights_im[add_word_idx,:] = transfer_weights_im
  if transfer_embed:
    embed_weights[add_word_idx, :] = embed
  if im_bias:
    predict_bias_im[add_word_idx] = transfer_bias_im

  for wi, close_word in enumerate(close_words_lm[add_word]['close_words']):
    close_word_idx = vocab_lines.index(close_word)
    predict_weights_im[close_word_idx,attribute_loc] = 0 
   
net.params[predict_lm][0].data[...] = predict_weights_lm
net.params[predict_lm][1].data[...] = predict_bias_lm
if transfer_embed:
  net.params['embedding'][0].data[...] = embed_weights
net.params['predict-im'][0].data[...] = predict_weights_im
if im_bias:
  net.params['predict-im'][1].data[...] = predict_bias_im
if transfer_embed:
  save_name = '%s.%s.embed.caffemodel' %(model_weights, save_tag)
else:
  save_name = '%s.%s.caffemodel' %(model_weights, save_tag)

net.save(save_name)

print close_words_im
print close_words_lm
print 'Saved to: %s' %save_name 
print 'Done.'
