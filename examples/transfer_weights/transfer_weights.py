import sys
sys.path.insert(0,'../../python/')
import caffe
import numpy as np
import copy
import pickle as pkl
import find_close_words

#model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_direct_predict_no_zebra_iter_110000'
#model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_attribute_JJ100_NN300_VB100_fc8_direct_no_caption_zebra_iter_110000'

#zebra
#model_weights = '/x/lisaanne/mrnn/snapshots_final/mrnn_attribute_JJ100_NN300_VB100_fc8_direct_no_zebra_captions_no_pretrain_lm_iter_110000'
model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_attributes_zebra_transfer0928_onlyZebraClassifier_iter_15000'
feature_word = 'zebra'
add_words = ['zebra', 'zebras']

#motorcycle
#model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_attributes_no_motorcycle_transfer0928_iter_110000'
#feature_word = 'motorcycle'
#add_words = ['motorcycle', 'motorcycles']

#pizza
#model_weights = '/x/lisaanne/mrnn/snapshots/mrnn_attributes_no_pizza_transfer0928_iter_110000'
#feature_word = 'pizza'
#add_words = ['pizza', 'pizzas']

model='../coco_attribute/mrnn_attributes_fc8-probs.direct.prototxt'
vocab_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'


new_words = {}
num_close_words=10
attributes = pkl.load(open('../coco_attribute/attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))

for word in add_words:
  new_words[word] = {}
  words, word_sims = find_close_words.find_close_words(add_words[0], add_words,num_words=num_close_words) 
  #new_words[word]['close_words'] = ['giraffe', 'cow', 'sheep']
  #new_words[word]['weights'] = [0.2712864, 0.22053696, 0.20719399]
  #new_words[word]['close_words'] = ['giraffe']
  #new_words[word]['weights'] = [1]
  replace_word = False
  w = -1
  while replace_word == False:
    if words[w] in attributes:
      replace_word = True
      new_words[word]['close_words'] = [words[w]]
      new_words[word]['weights'] = [1]
    w -= 1     
 
  new_words[word]['weights'] = new_words[word]['weights']/np.linalg.norm(new_words[word]['weights'])


net = caffe.Net(model, model_weights + '.caffemodel', caffe.TRAIN)

vocab_lines = open(vocab_file, 'rb').readlines()
vocab_lines = [v.strip() for v in vocab_lines]
vocab_lines = ['<EOS>'] + vocab_lines

predict_weights_lm = copy.deepcopy(net.params['predict-lm'][0].data)
predict_bias_lm = copy.deepcopy(net.params['predict-lm'][1].data)
predict_weights_im = copy.deepcopy(net.params['predict-im'][0].data)
#predict_bias_im = copy.deepcopy(net.params['predict-im'][1].data)
attribute_loc = attributes.index(feature_word)

for new_word in new_words.keys():
  new_word_idx = vocab_lines.index(new_word)
  transfer_weights_lm = np.ones((predict_weights_lm.shape[1],))*0
  transfer_bias_lm = 0
  transfer_weights_im = np.ones((predict_weights_im.shape[1],))*0

  for wi, close_word in enumerate(new_words[new_word]['close_words']):
    close_word_idx = vocab_lines.index(close_word)

    transfer_weights_lm += predict_weights_lm[close_word_idx,:]*new_words[new_word]['weights'][wi]
    transfer_bias_lm += predict_bias_lm[close_word_idx]*new_words[new_word]['weights'][wi]
    transfer_weights_im += predict_weights_im[close_word_idx,:]*new_words[new_word]['weights'][wi]
#    transfer_weights_lm = np.maximum(predict_weights_lm[close_word_idx,:], transfer_weights_lm)
#    transfer_bias_lm = np.maximum(predict_bias_lm[close_word_idx], transfer_bias_lm)
#    transfer_weights_im = np.maximum(predict_weights_im[close_word_idx,:], transfer_weights_im)

  #transfer new word feature to new word index like transferring old word feature to old word index
#  predict_weights_lm[new_word_idx,:] = transfer_weights_lm/num_close_words 
#  predict_bias_lm[new_word_idx] = transfer_bias_lm/num_close_words
#  predict_weights_im[new_word_idx,:] = transfer_weights_im/num_close_words
  predict_weights_lm[new_word_idx,:] = transfer_weights_lm
  predict_bias_lm[new_word_idx] = transfer_bias_lm
  predict_weights_im[new_word_idx,:] = transfer_weights_im

  predict_weights_im[new_word_idx, attribute_loc] = 0 

  count_close_attributes = 0
  for wi, close_word in enumerate(new_words[new_word]['close_words']):
    #cross attribute weights
    predict_weights_im[close_word_idx, attribute_loc] = 0
    
    close_word_attribute_loc = None
    if not close_word in attributes or close_word[:-1] in attributes:  #have to deal with plurals which is a bit of a pain
      if close_word[:-1] in attributes:
        if close_word[-1] == 's':
          close_word_attribute_loc = attributes.index(close_word[:-1])
    else:
      close_word_attribute_loc = attributes.index(close_word)

    if close_word_attribute_loc:
      predict_weights_im[new_word_idx, close_word_attribute_loc] = 0 
      predict_weights_im[new_word_idx, attribute_loc] += predict_weights_im[close_word_idx, close_word_attribute_loc] 
      count_close_attributes += 1
  predict_weights_im[new_word_idx, attribute_loc] /= count_close_attributes 
   
net.params['predict-lm'][0].data[...] = predict_weights_lm
net.params['predict-lm'][1].data[...] = predict_bias_lm
net.params['predict-im'][0].data[...] = predict_weights_im
#net.params['predict-im'][1].data[...] = predict_bias_im
net.save('%s.transfer_closest_attribute.caffemodel' %model_weights)

print 'Done.'
