import sys
sys.path.insert(0,'../../python')
sys.path.insert(0, '../word_similarity/')
import caffe
from init import *
import numpy as np
import pickle as pkl
from w2vDist import *

adapt_embed = False 
mag = True
eightyk = False


num_close_words_im_list = [1, 1]
num_close_words_lm_list = [1, 3]
rm_word_base_list = ['bottle', 'bus', 'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']

for num_close_words_im, num_close_words_lm in zip(num_close_words_im_list, num_close_words_lm_list):
  for rm_word_base in rm_word_base_list:

    save_tag = '%s.da_av_im%d_lm%d' %(rm_word_base, num_close_words_im, num_close_words_lm)
    
    #zebra
    if rm_word_base == 'zebra':
      add_words = {'words': ['zebra', 'zebras'], 'classifiers': ['zebra', 'zebra'], 'illegal_words': ['zebra']}
    #bus
    if rm_word_base == 'bus':
      add_words = {'words': ['bus', 'busses'], 'classifiers': ['bus', 'bus'], 'illegal_words': ['bus']}
    if rm_word_base == 'pizza':
      add_words = {'words': ['pizza', 'pizzas'], 'classifiers': ['pizza', 'pizza'], 'illegal_words': ['pizza']}
    if rm_word_base == 'suitcase':
      add_words = {'words': ['suitcase', 'suitcases', 'luggage', 'luggages'], 'classifiers': ['suitcase', 'suitcase', 'luggage', 'luggage'], 'illegal_words': ['luggage', 'suitcase']}
    if rm_word_base == 'bottle':
      add_words = {'words': ['bottle', 'bottles'], 'classifiers': ['bottle', 'bottle'], 'illegal_words': ['bottle']}
    if rm_word_base == 'couch':
       add_words = {'words': ['couch', 'couches'], 'classifiers': ['couch', 'couch'], 'illegal_words': ['couch']}
    if rm_word_base == 'microwave':
      add_words = {'words': ['microwave', 'microwaves'], 'classifiers': ['microwave', 'microwave'], 'illegal_words': ['microwave']}
    if rm_word_base == 'racket':
      add_words = {'words': ['racket', 'rackets', 'racquet', 'racquets'], 'classifiers': ['racket', 'racket', 'racquet', 'racquet'], 'illegal_words': ['racket', 'racquet']}
    if rm_word_base == 'all':
      add_words = {'words': ['zebra', 'zebras', 'bus', 'busses', 'pizza', 'pizzas', 'suitcase', 'suitcases', 'luggage', 'luggages','bottle', 'bottles', 'couch', 'couches', 'microwave', 'microwaves', 'racket', 'rackets', 'racquet', 'racquets'], 'classifiers': ['zebra', 'zebra', 'bus', 'bus', 'pizza', 'pizza', 'suitcase', 'suitcase', 'luggage', 'luggage', 'bottle', 'bottle', 'couch', 'couch', 'microwave', 'microwave', 'racket', 'racket', 'racquet', 'racquet'], 'illegal_words': ['zebra', 'bus', 'pizza', 'luggage', 'suitcase','bottle', 'couch', 'microwave', 'racket', 'racquet']}
    
    model = 'mrnn_attributes_fc8.direct.from_features.wtd.ft.prototxt' #the wtd has all the parameters we care about and will take up less memory
    if not eightyk: 
      model_weights_lm = pretrained_lm + 'mrnn.direct_iter_110000'
    else:
      model_weights_lm = pretrained_lm + 'mrnn.lm.direct_imtextyt_lr0.01_iter_120000'
      #model_weights_lm = pretrained_lm + 'mrnn.lm.direct_surf_lr0.01_iter_120000'
      #model_weights_lm = pretrained_lm + 'mrnn.lm.direct_wikisent_lr0.01_iter_120000'
    
    #model_weights_trained = '/z/lisaanne/snapshots_caption_models/attributes_jj100_nn300_vb100_zebra_cocoimages_captions_ftlmpretrain_iter_110000'
    #model_weights_trained = '/z/lisaanne/snapshots_caption_models/attributes_jj100_nn300_vb100_zebra_cocoimages_captions_ftlmpretrain_pretrainlm50_iter_110000'
    model_weights_trained = '/y/lisaanne/mrnn_direct/snapshots/attributes_JJ100_NN300_VB100_eightClusters_captions_imagenetImages_1026_ftLM_pretrain50k_1031_iter_10000'
    
    
    net_lm = caffe.Net(model, model_weights_lm + '.caffemodel', caffe.TEST)
    net_trained = caffe.Net(model, model_weights_trained + '.caffemodel', caffe.TEST)
    
    if not eightyk:
      vocab = open('../coco_caption/h5_data/buffer_100/vocabulary.txt').readlines()
    else:
      vocab = open('/x/lisaanne/pretrained_lm/yt_coco_surface_80k_vocab.txt').readlines()
    vocab = [v.strip() for v in vocab]
    vocab = ['eos'] + vocab
    
    im_bias = False
    if len(net_trained.params['predict-im']) > 1:
      im_bias = True
    
    attributes = pkl.load(open('../coco_attribute/attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))
    
    #create word to vec
    W2V = w2v()
    W2V.readVectors()
    W2V.reduce_vectors(attributes, '-n')
    
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
      
    
