import sys
sys.path.insert(0,'../../python/')
sys.path.insert(0, '../')
from utils.python_utils import *
from w2vDist import *
import caffe
import numpy as np
import copy
import pickle as pkl
import hickle as hkl
from nltk.corpus import wordnet as wn
from utils.config import *
import pdb

class closeness_embedding(object):
  
  def __init__(self, attributes):
    self.W2V = w2v()
    self.W2V.readVectors()
    self.W2V.reduce_vectors(attributes, '-n')

  def __call__(self, word):
    return self.W2V.findClosestWords(word) 

class closeness_embedding_refined(object):
  pass

class transfer_net(object):

  def __init__(self, model, model_weights, orig_attributes, all_attributes, vocab):
    self.model = model
    self.model_weights = model_weights
    self.net = caffe.Net(self.model, caption_weights_root + self.model_weights + '.caffemodel', caffe.TRAIN)
    self.orig_attributes = open_txt(orig_attributes)
    self.all_attributes = open_txt(all_attributes)
    self.new_attributes = list(set(self.all_attributes) - set(self.orig_attributes))
    self.vocab = open_txt(vocab)
    self.vocab = ['<EOS>'] + self.vocab
 
  def direct_transfer(self, words, classifiers, closeness_metric, log=False, predict_lm = 'predict-lm', predict_im = 'predict-im', num_transfer=1, orig_net_weights=''): 
   metric = eval(closeness_metric)(self.all_attributes)
   save_tag = '%s_%s' %(words.split('/')[-1], closeness_metric)
   if log: log_file = 'outfiles/transfer/%s_%s_direct.out' %(self.model_weights, words.split('/')[-1])
   if log: log_write = open(log_file, 'w')

   words = open_txt(words)
   classifiers = open_txt(classifiers)

   illegal_words = classifiers + self.new_attributes
   illegal_idx = [self.all_attributes.index(illegal_word) for illegal_word in illegal_words]
     
   if len(self.net.params[predict_im]) > 1:
     im_bias = True
   else: 
     im_bias = False

   close_words = {}
   for word, classifier in zip(words, classifiers):
     word_sims = metric(classifier)
     for illegal_id in illegal_idx:
       word_sims[illegal_id] = -100000

     close_words[word] = self.all_attributes[np.argsort(word_sims)[-1]]
     
     t_word_string =  "Transfer word for %s is %s." %(word, close_words[word])
     print t_word_string
     if log: log_write.writelines('%s\n' %t_word_string) 

   predict_weights_lm = copy.deepcopy(self.net.params[predict_lm][0].data)
   predict_bias_lm = copy.deepcopy(self.net.params[predict_lm][1].data)
   predict_weights_im = copy.deepcopy(self.net.params[predict_im][0].data)
   if im_bias:
     predict_bias_im = copy.deepcopy(self.net.params[predict_im][1].data)
   
   for word, classifier in zip(words, classifiers):
     word_idx = self.vocab.index(word)
     attribute_idx = self.all_attributes.index(classifier)
     transfer_word_idx = self.vocab.index(close_words[word])     
     transfer_attribute_idx = self.all_attributes.index(close_words[word])     

     transfer_weights_lm = np.ones((predict_weights_lm.shape[1],))*0
     transfer_bias_lm = 0
     transfer_weights_im = np.ones((predict_weights_im.shape[1],))*0
     if im_bias:
       transfer_bias_im = 0  
 
     transfer_weights_lm += predict_weights_lm[transfer_word_idx,:]
     transfer_bias_lm += predict_bias_lm[transfer_word_idx]
     transfer_weights_im += predict_weights_im[transfer_word_idx,:]
     if im_bias:    
       transfer_bias_im += predict_bias_im[transfer_word_idx]
 
     #Take care of classifier cross terms
     transfer_weights_im[attribute_idx] = predict_weights_im[transfer_word_idx, transfer_attribute_idx]
     transfer_weights_im[transfer_attribute_idx] = 0
    
     predict_weights_lm[word_idx,:] = transfer_weights_lm
     predict_bias_lm[word_idx] = transfer_bias_lm
     predict_weights_im[word_idx,:] = transfer_weights_im
     if im_bias:
       predict_bias_im[word_idx] = transfer_bias_im  
 
     predict_weights_im[transfer_word_idx,attribute_idx] = 0 
   
   self.net.params[predict_lm][0].data[...] = predict_weights_lm
   self.net.params[predict_lm][1].data[...] = predict_bias_lm
   self.net.params[predict_im][0].data[...] = predict_weights_im
   if im_bias:
     self.net.params[predict-im][1].data[...] = predict_bias_im
   self.net.save('%s%s.%s.caffemodel' %(caption_weights_root, self.model_weights, save_tag))
   
   save_string = 'Saved to: %s%s.%s.caffemodel' %(caption_weights_root, self.model_weights, save_tag) 
   print save_string 
   if log: log_write.writelines('%s\n' %(save_string)) 
   if log: log_write.close()
   if log: print 'Log file saved to %s.' %log_file 

  def delta_transfer(self, words, classifiers, closeness_metric, log=False, predict_lm = 'predict', predict_im = 'predict-im', num_transfer=3, orig_net_weights=''): 

   metric = eval(closeness_metric)(self.all_attributes)
   save_tag = '%s_%s' %(words.split('/')[-1], closeness_metric)
   if log: log_file = 'outfiles/transfer/%s_%s_%d_delta.out' %(self.model_weights, words.split('/')[-1], num_transfer)
   if log: log_write = open(log_file, 'w')

   orig_net = caffe.Net(self.model, caption_weights_root + orig_net_weights + '.caffemodel', caffe.TRAIN)

   words = open_txt(words)
   classifiers = open_txt(classifiers)

   illegal_words = classifiers + self.new_attributes
   illegal_idx = [self.all_attributes.index(illegal_word) for illegal_word in illegal_words]
     
   if len(self.net.params[predict_im]) > 1:
     im_bias = True
   else: 
     im_bias = False

   close_words = {}
   for word, classifier in zip(words, classifiers):
     word_sims = metric(classifier)
     for illegal_id in illegal_idx:
       word_sims[illegal_id] = -100000

     close_words[word] = [self.all_attributes[np.argsort(word_sims)[-1*i]] for i in range(1,num_transfer+1)]
     
     t_word_string =  "Transfer words for %s are %s." %(word, close_words[word])
     print t_word_string
     if log: log_write.writelines('%s\n' %t_word_string) 

   ft_weights_lm = copy.deepcopy(self.net.params[predict_lm][0].data)
   orig_weights_lm = copy.deepcopy(orig_net.params[predict_lm][0].data)
   ft_bias_lm = copy.deepcopy(self.net.params[predict_lm][1].data)
   orig_bias_lm = copy.deepcopy(orig_net.params[predict_lm][1].data)
   ft_weights_im = copy.deepcopy(self.net.params[predict_im][0].data)
   if im_bias:
     ft_bias_im = copy.deepcopy(self.net.params[predict_im][1].data)
   
   for word, classifier in zip(words, classifiers):
     word_idx = self.vocab.index(word)
     attribute_idx = self.all_attributes.index(classifier)
     transfer_weights_lm = np.ones((ft_weights_lm.shape[1],))*0
     transfer_bias_lm = 0 
     transfer_weights_im = np.ones((ft_weights_im.shape[1],))*0
     if im_bias:
       transfer_bias_im = 0  

     for close_word in close_words[word]:
       transfer_word_idx = self.vocab.index(close_word)     
       transfer_attribute_idx = self.all_attributes.index(close_word)      
   
       transfer_weights_lm += ft_weights_lm[transfer_word_idx,:] - orig_weights_lm[transfer_word_idx,:]
       transfer_bias_lm += ft_bias_lm[transfer_word_idx] - orig_bias_lm[transfer_word_idx]
 
     #Take care of classifier cross terms
     transfer_word_idx = self.vocab.index(close_words[word][0])     
     transfer_attribute_idx = self.all_attributes.index(close_words[word][0])     
     transfer_weights_im += ft_weights_im[transfer_word_idx,:]
     if im_bias:    
       transfer_bias_im += ft_bias_im[transfer_word_idx]
     transfer_weights_im[attribute_idx] = ft_weights_im[transfer_word_idx, transfer_attribute_idx]
     transfer_weights_im[transfer_attribute_idx] = 0
    
     ft_weights_lm[word_idx,:] += transfer_weights_lm/num_transfer
     ft_bias_lm[word_idx] += transfer_bias_lm/num_transfer
     ft_weights_im[word_idx,:] = transfer_weights_im
     if im_bias:
       ft_bias_im[word_idx] = transfer_bias_im  
 
     ft_weights_im[transfer_word_idx,attribute_idx] = 0 
   
   self.net.params[predict_lm][0].data[...] = ft_weights_lm
   self.net.params[predict_lm][1].data[...] = ft_bias_lm
   self.net.params[predict_im][0].data[...] = ft_weights_im
   if im_bias:
     self.net.params[predict-im][1].data[...] = ft_bias_im
   self.net.save('%s%s.%s_delta_%d.caffemodel' %(caption_weights_root, self.model_weights, save_tag, num_transfer))
   
   save_string = 'Saved to: %s%s.%s_delta_%d.caffemodel' %(caption_weights_root, self.model_weights, save_tag, num_transfer) 
   print save_string 
   if log: log_write.writelines('%s\n' %(save_string)) 
   if log: log_write.close()
   if log: print 'Log file saved to %s.' %log_file 

















 
