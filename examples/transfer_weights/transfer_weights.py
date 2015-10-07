import sys
sys.path.insert(0,'../../python/')
import caffe
import numpy as np
import copy
import pickle as pkl
import find_close_words

save_tag = 'transfer_closest_im_lm.commonWords1000'
num_close_words_im = 1
num_close_words_lm = 1

#zebra
#model_weights = '/x/lisaanne/mrnn/snapshots_final/mrnn_attribute_JJ100_NN300_VB100_fc8_direct_no_zebra_captions_no_pretrain_lm_iter_110000'
#model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_attributes_zebra_transfer0928_onlyZebraClassifier_iter_15000'
model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_attributes_zebra_transfer0928_onlyZebraClassifier_ft_all_iter_110000'
add_words = {'words': ['zebra', 'zebras'], 'classifiers': ['zebra', 'zebra'], 'illegal_words': ['zebra', 'zebras']}

#motorcycle
#model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_attributes_no_motorcycle_transfer0928_iter_110000'
#model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_attributes_zebra_transfer0928_onlyMotorcycleClassifier_ft_all_iter_90000'
#add_words = {'words':['motorcycle', 'motorcycles', 'motor', 'cycle', 'cycles'], 'classifiers':['motorcycle', 'motorcycle', 'motorcycle', 'motorcycle', 'motorcycle'], 'illegal_words': ['motorcycle', 'motorcycles', 'motor', 'cycle', 'cycles']}

#pizza
#model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_attributes_no_pizza_transfer0928_iter_110000'
#feature_word = 'pizza'
#add_words = ['pizza', 'pizzas']

model='../coco_attribute/mrnn_attributes_fc8-probs.direct.prototxt'
net = caffe.Net(model, model_weights + '.caffemodel', caffe.TRAIN)
att_weights = copy.deepcopy(net.params['fc8-attributes'][0].data)

def closeness_embedding(new_word, illegal_words, num_close_words):
  words, word_sims = find_close_words.find_close_words(new_word, illegal_words, num_close_words)
  return words, word_sims

def closeness_classifier_weights(new_word, illegal_words, num_close_words):
  new_word_idx = attributes.index(new_word)
  illegal_words_idx = [attributes.index(iw) for iw in illegal_words if iw in attributes]
  norm_atts = att_weights/np.tile(np.linalg.norm(att_weights, axis=1), (4096,1)).T
  euc_dist = np.linalg.norm(norm_atts[new_word_idx,:] - norm_atts, axis=1) 
  for iw in illegal_words_idx: euc_dist[iw] = 10001
  return [attributes[ed] for ed in np.argsort(euc_dist)[:num_close_words]], np.sort(euc_dist)[:num_close_words]

closeness_metric_lm = closeness_embedding 
closeness_metric_im = closeness_classifier_weights

close_words_im = {}
close_words_lm = {}
attributes = pkl.load(open('../coco_attribute/attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))

for aw, word in enumerate(add_words['words']):
  close_words_im[word] = {}
  words, word_sims = closeness_metric_im(add_words['classifiers'][aw], add_words['illegal_words'], num_close_words_im) 
  close_words_im[word]['close_words'] = words
  close_words_im[word]['weights'] = word_sims/np.linalg.norm(word_sims, 1)

  close_words_lm[word] = {}
  words, word_sims = closeness_metric_lm(word, add_words['words'],num_close_words_lm) 
  close_words_lm[word]['close_words'] = words
  close_words_lm[word]['weights'] = word_sims/np.linalg.norm(word_sims, 1)

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
