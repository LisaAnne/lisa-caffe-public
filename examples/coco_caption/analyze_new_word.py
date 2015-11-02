import numpy as np
import json
import sys
sys.path.insert(0, '../coco_caption/')
import retrieval_experiment
import sys
import os
import argparse
import re
import argparse

#This script will run all the following tests:
	#(1) Sentence generation for beam1 and beam3
	#(2) F1-score

#example input: python -m pdb analyze_new_word.py augment_train_noZebra_vocab8749_iter_110000 /models/bvlc_reference_caffenet/deploy.prototxt /examples/coco_caption/lrcn_word_to_preds.deploy.prototxt augment_train_noZebra_ 1 1

print " I am changing this file!"

home_dir = '../../'
retrieval_cache_home = '/x/lisaanne/retrieval_cache/'

def read_json(json_file):
  t_json = open(json_file).read()
  return json.loads(t_json) 

def split_sent(sent):
  sent = sent.lower()
  sent = re.sub('[^A-Za-z0-9\s]+','', sent)
  return sent.split()

parser = argparse.ArgumentParser()
parser.add_argument("--trained_model",type=str)
parser.add_argument("--image_model",type=str)
parser.add_argument("--language_model",type=str)
parser.add_argument("--rm_word_base",type=str,default='zebra')
parser.add_argument("--feature_dir_h5",type=str,default=None)
parser.add_argument("--feats_bool_in",type=bool,default=True)
parser.add_argument("--eightyK",type=str,default='True')
parser.add_argument("--gpu",type=int,default=0)


args = parser.parse_args()

print args.feature_dir_h5

trained_model = args.trained_model.split(',')
image_model = args.image_model
language_model = args.language_model
rm_word_base = args.rm_word_base
feature_dir_h5 = args.feature_dir_h5
feats_bool_in = args.feats_bool_in
eightyK = eval(args.eightyK)

rm_word_base_list = []
rm_words_list = []

if rm_word_base == 'bus' or rm_word_base == 'eightCluster1':
  rm_words = ['bus', 'busses']
  rm_word_base_list.append('bus')
  rm_words_list.append(rm_words)
if rm_word_base == 'bottle' or rm_word_base == 'eightCluster1':
  rm_words = ['bottle', 'bottles']
  rm_word_base_list.append('bottle')
  rm_words_list.append(rm_words)
if rm_word_base == 'couch' or rm_word_base == 'eightCluster1':
  rm_words = ['couch', 'couches', 'sofa', 'sofas']
  rm_word_base_list.append('couch')
  rm_words_list.append(rm_words)
if rm_word_base == 'microwave' or rm_word_base == 'eightCluster1':
  rm_words = ['microwave', 'microwaves']
  rm_word_base_list.append('microwave')
  rm_words_list.append(rm_words)
if rm_word_base ==  'pizza' or rm_word_base == 'eightCluster1':
  rm_words = ['pizza', 'pizzas']
  rm_word_base_list.append('pizza')
  rm_words_list.append(rm_words)
if rm_word_base == 'racket' or rm_word_base == 'eightCluster1':
  rm_words = ['racket', 'rackets']
  rm_word_base_list.append('racket')
  rm_words_list.append(rm_words)
if rm_word_base ==  'suitcase' or rm_word_base == 'eightCluster1':
  rm_words = ['luggage', 'luggages', 'suitcase', 'suitcases']
  rm_word_base_list.append('suitcase')
  rm_words_list.append(rm_words)
if rm_word_base == 'zebra' or rm_word_base == 'eightCluster1':
  rm_words = ['zebra', 'zebras']
  rm_word_base_list.append('zebra')
  rm_words_list.append(rm_words)

if rm_word_base ==  'giraffe':
  rm_words = ['giraffe', 'giraffe', 'giraffes', 'girafee', 'giraffee', 'giraff']
if rm_word_base == 'motorcycle':
  rm_words = ['motor', 'motors', 'cycle', 'cycles', 'motorcycle', 'motorcycles']


tag = '' 
full_vocabulary_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'  
vocab = 'vocabulary'
if eightyK:
  full_vocabulary_file = '../coco_caption/h5_data/buffer_100/vocabulary80k.txt'  
  vocab = 'vocabulary80k'
beam1 = True
beam3 = False


#do sentence generation
beam1_json_path = retrieval_cache_home + 'val_val/all_ims/%s/beam1/generation_result.json' %('_'.join(trained_model))
always_recompute = True
if beam1:
  if not os.path.exists(beam1_json_path) or always_recompute:
    experiment = {'type': 'generation'}
    retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name='val_val', vocab=vocab, precomputed_feats=None, feats_bool_in=feats_bool_in, precomputed_h5=feature_dir_h5, experiment=experiment, prev_word_restriction=True, gpu=args.gpu)
  else:
    experiment = {'type': 'score_generation', 'json_file': beam1_json_path} 
    retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name='val_val', vocab=vocab, precomputed_feats=feature_dir, feats_bool_in=False, experiment=experiment, prev_word_restriction=True)

if beam3:
  if not os.path.exists(retrieval_cache_home + '%s/all_ims/%s/beam3/generation_result.json' %(test_set, trained_model)):
    experiment = {'type': 'generation', 'beam_size': 3}
    retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name='val_val', vocab=vocab, precomputed_feats=feature_dir, feats_bool_in=False, experiment=experiment)

def eval_generation(generated_sentences, gt_file, test_set):
  generated_sentences_json = read_json(generated_sentences)
  gt_json = read_json(gt_file)
  image_ids = list(np.unique([i['id'] for i in gt_json['images']]))  
  gen_novel = [g for g in generated_sentences_json if g['image_id'] in image_ids]
  tmp_json = 'gt_file_tmp.%s.%s.json' %(trained_model[-1].split('/')[-1], rm_word_base)
  with open(tmp_json, 'w') as outfile:
    json.dump(gen_novel, outfile)
  experiment = {'type': 'score_generation', 'json_file': tmp_json, 'read_file': True} 
  retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name=test_set, vocab=vocab, feats_bool_in=False, experiment=experiment)
  os.remove(tmp_json)

for rm_word_base, rm_words in zip(rm_word_base_list, rm_words_list):

  print rm_word_base
  print rm_words

  gt_novel_json = '../../data/coco/coco/annotations/captions_split_set_%s_val_val_novel2014.json' %rm_word_base
  gt_train_json = '../../data/coco/coco/annotations/captions_split_set_%s_val_val_train2014.json' %rm_word_base
  set_novel = 'split_set_%s_val_val_novel' %rm_word_base
  set_train = 'split_set_%s_val_val_train' %rm_word_base
  
  print 'Scores for novel captions in val set for word %s:\n' %rm_word_base
  eval_generation(beam1_json_path, gt_novel_json, set_novel) 
  print 'Scores for train captions in val set for word %s:\n' %rm_word_base
  eval_generation(beam1_json_path, gt_train_json, set_train) 
  
  
  #compute f1-score for missing words using generated captions
  
  def compute_f1_score(generated_json_path, gt_file_novel):
    generated_sentences_json = read_json(generated_json_path)
    gt_json = read_json(gt_file_novel)
    image_ids = list(np.unique([i['id'] for i in gt_json['images']]))  
    gen_novel = [g for g in generated_sentences_json if g['image_id'] in image_ids]
    gen_train = [g for g in generated_sentences_json if g['image_id'] not in image_ids]
    set_rm_words = set(rm_words)
    #true positive are sentences that contain match words and should
    tp = sum([1 for c in gen_novel if len(set_rm_words.intersection(set(split_sent(c['caption'])))) > 0]) 
    #false positive are sentences that contain match words and should not
    fp = sum([1 for c in gen_train if len(set_rm_words.intersection(set(split_sent(c['caption'])))) > 0])
    #false positive are sentences that do not contain match words and should
    fn = sum([1 for c in gen_novel if len(set_rm_words.intersection(set(split_sent(c['caption'])))) == 0 ])
   
    #precision = tp/(tp+fp)
    if tp > 0:  
      precision = float(tp)/(tp+fp) 
      #recall = tp/(tp+fn)
      recall = float(tp)/(tp+fn)
      #f1 = 2* (precision*recall)/(precision+recall)
      return 2*(precision*recall)/(precision+recall)
    else:
      return 0.
  
  f_score = compute_f1_score(beam1_json_path, gt_novel_json)
  print 'F1 score for beam1 and objects %s is: %f' %(rm_word_base, f_score)

##fill in the blank metrics
#experiment = {'type': 'madlib', 'fill_words': ['zebra', 'zebras'], 'cooccur_words': [[]]}
#all_mean_index, all_mean_prob, all_top_words = retrieval_experiment.main(trained_model, image_model, language_model, test_set, vocabulary_file, experiment)
#
#print "Madlib metrics:"
#all_metrics = [{'out_words': 'Mean index for fw %s and cw %%s: ', 'metric': all_mean_index}, \
#               {'out_words': 'Mean probability for fw %s and cw %%s: ', 'metric': all_mean_prob}, \
#               {'out_words': 'Other common words for fw %%s and cw %%s: ', 'metric': all_top_words}]
#for metric in all_matrics:
#  for fw in experiment['fill_words']: 
#    for cw in experiment['cooccur_words']:
#      print '%s %s' %(metric['out_words'] %(fw, cw), metric['metric'])
#print '\n'
#
##look at difference in vocabulary words
#baseline_vocab_file = open('h5_data/buffer_100/vocabualry.txt','r')
#comp_vocab_file = open('h5_data/buffer_100/%s.txt' %vocabulary_file)
#
#baseline_words = [line.strip('\n') for line in baseline_vocab_file.readline()]
#comp_words = [line.strip('\n') for line in comp_vocab_file.readline()]
#
#missing_words = [word for word in comp_words if word not in baseline_words]
#print 'When excluding words: %s the following words also get cut from the vocabulary:\n %s' %(', '.join(a), ' '.join(missing_words))



