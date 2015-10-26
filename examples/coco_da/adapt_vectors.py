import sys
sys.path.insert(0,'../../python')
import caffe

import numpy as np
import pickle as pkl

w = ''
save_tag = 'experiment_da_dlm_debug%s' %w

model = 'mrnn_attributes_fc8-probs.direct_attributes.wtd.prototxt' #The wtd has all the parameters we care about and will take up less memory

model_weights_lm = '/z/lisaanne/pretrained_lm/mrnn.direct_iter_110000'
#model_weights_lm = '/x/lisaanne/pretrained_lm/mrnn.lm.direct_surf_lr0.01_iter_120000'
model_weights_im = 'attributes_JJ100_NN300_VB100_zebra_iter_50000'

#model_weights_trained = '/x/lisaanne/coco_caption/snapshots/coco_da_train_direct_attributes_ftPredict_pretrainAttributes_zebraClassifier_pretrain50k_iter_110000'
model_weights_trained= '/z/lisaanne/snapshots_caption_models/attributes_JJ100_NN300_VB100_zebra_cocoImages_captions_ftLMPretrain_iter_110000'
#model_weights_trained = '/x/lisaanne/coco_caption/snapshots/coco_da_train_direct_attributes_ftPredictLM_pretrainAttributes_zebraClassifier_80k_run2_iter_210000'  

net_lm = caffe.Net(model, model_weights_lm + '.caffemodel', caffe.TEST)
net_im = caffe.Net(model, model_weights_im + '.caffemodel', caffe.TEST)
net_trained = caffe.Net(model, model_weights_trained + '.caffemodel', caffe.TEST)

vocab = open('../coco_caption/h5_data/buffer_100/vocabulary.txt').readlines()
#vocab = open('/x/lisaanne/pretrained_lm/yt_coco_surface_80k_vocab.txt').readlines()
vocab = [v.strip() for v in vocab]
vocab = ['EOS'] + vocab

add_words = ['zebra', 'zebras']
add_word_classifiers = ['zebra', 'zebra']
transfer_words_lm = [['giraffe'], ['giraffe']]
transfer_words_im = [['giraffe'], ['giraffe']]

attributes = pkl.load(open('../coco_attribute/attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))

for add_word, add_word_classifier, transfer_word_lm, transfer_word_im in zip(add_words, add_word_classifiers, transfer_words_lm, transfer_words_im):
  #transfer im
  #collect delta on lm

  add_word_idx = vocab.index(add_word)
  delta_weights = np.zeros(net_trained.params['predict'][0].data.shape[1],)
  delta_bias = 0
  for twl in transfer_word_lm:
    transfer_word_lm_idx = vocab.index(twl)
    delta_transfer_weights = net_trained.params['predict'][0].data[transfer_word_lm_idx,:] - net_lm.params['predict'][0].data[transfer_word_lm_idx,:]
    delta_transfer_bias = net_trained.params['predict'][1].data[transfer_word_lm_idx] - net_lm.params['predict'][1].data[transfer_word_lm_idx]
    delta_weights += delta_transfer_weights
    delta_bias += delta_transfer_bias

  net_trained.params['predict'][0].data[add_word_idx,:] += delta_transfer_weights/len(transfer_word_lm)
  net_trained.params['predict'][1].data[add_word_idx] += delta_transfer_bias/len(transfer_word_lm)
  
#  net_trained.params['predict'][0].data[add_word_idx,:] = net_trained.params['predict'][0].data[transfer_word_lm_idx,:]
#  net_trained.params['predict'][1].data[add_word_idx] = net_trained.params['predict'][1].data[transfer_word_lm_idx]

  #transfer im (same as before -- just do direct transfer)

  weights_im_transfer = np.zeros(net_trained.params['predict-im'][0].data.shape[1],)
  bias_im_transfer = 0
 
  for twi in transfer_word_im:
    transfer_word_im_idx = vocab.index(twi)
    weights_im_transfer += net_trained.params['predict-im'][0].data[transfer_word_im_idx,:]
    #bias_im_transfer += net_trained.params['predict-im'][1].data[transfer_word_im_idx]
    
  net_trained.params['predict-im'][0].data[add_word_idx,:] = weights_im_transfer/len(transfer_word_im)
#  net_trained.params['predict-im'][1].data[add_word_idx] = bias_im_transfer/len(transfer_word_im)

  add_attribute_idx = attributes.index(add_word_classifier)  
  self_correlation = 0
  for twi in transfer_word_im:
    transfer_attribute_idx = attributes.index(twi)  
    #co-correlation 
    self_correlation += net_trained.params['predict-im'][0].data[transfer_word_im_idx, transfer_attribute_idx]
    net_trained.params['predict-im'][0].data[add_word_idx, transfer_attribute_idx] = 0 
    net_trained.params['predict-im'][0].data[transfer_word_im_idx, add_attribute_idx] = 0 

  net_trained.params['predict-im'][0].data[add_word_idx, add_attribute_idx] = self_correlation/len(transfer_word_im) 
  

net_trained.save('%s.%s.caffemodel' %(model_weights_trained, save_tag))
print 'Saved at: %s.%s.caffemodel' %(model_weights_trained, save_tag)
  

