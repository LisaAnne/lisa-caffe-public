import numpy as np
import json
import sys
sys.path.insert(0, '../coco_caption/')
import retrieval_experiment
import sys
import os
import argparse
import re

#This script will run all the following tests:
	#(1) Sentence generation for beam1 and beam3
	#(2) F1-score

#example input: python -m pdb analyze_new_word.py augment_train_noZebra_vocab8749_iter_110000 /models/bvlc_reference_caffenet/deploy.prototxt /examples/coco_caption/lrcn_word_to_preds.deploy.prototxt augment_train_noZebra_ 1 1

home_dir = '../../'
retrieval_cache_home = '/x/lisaanne/retrieval_cache/'

def read_json(json_file):
  t_json = open(json_file).read()
  return json.loads(t_json) 

def split_sent(sent):
  sent = sent.lower()
  sent = re.sub('[^A-Za-z0-9\s]+','', sent)
  return sent.split()


trained_model = sys.argv[1]
trained_model = trained_model.split(',')
image_model = sys.argv[2]
language_model = sys.argv[3]
rm_word_base = sys.argv[4]
if len(sys.argv) > 5:
  feature_dir_h5 = sys.argv[5]
  feats_bool_in = True
else: 
  feature_dir_h5 = None
  feats_bool_in = False
if len(sys.argv) > 6:
  feature_dir = sys.argv[6]
else: 
  feature_dir = None

if rm_word_base == 'zebra':
  rm_words = ['zebra', 'zebras']
if rm_word_base ==  'giraffe':
  rm_words = ['giraffe', 'giraffe', 'giraffes', 'girafee', 'giraffee', 'giraff']
if rm_word_base == 'motorcycle':
  rm_words = ['motor', 'motors', 'cycle', 'cycles', 'motorcycle', 'motorcycles']
if rm_word_base ==  'pizza':
  rm_words = ['pizza', 'pizzas']
if rm_word_base ==  'suitcase':
  rm_words = ['luggage', 'luggages', 'suitcase', 'suitcases']
if rm_word_base == 'bottle':
  rm_words = ['bottle', 'bottles']
if rm_word_base == 'couch':
  rm_words = ['couch', 'couches', 'sofa', 'sofas']
if rm_word_base == 'microwave':
  rm_words = ['microwave', 'microwaves']
if rm_word_base == 'racket':
  rm_words = ['racket', 'rackets']
if rm_word_base == 'bus':
  rm_words = ['bus', 'busses']
if rm_word_base == 'rm_eightCluster':
  rm_words = ['luggage', 'luggages', 'suitcase', 'suitcases', 'bottle', 'bottles', 'couch', 'couches', 'sofa', 'so    fas', 'microwave', 'microwaves', 'rackett', 'racket', 'raquet', 'rackets',  'bus', 'buses', 'busses', 'pizza', 'pizz    as', 'zebra', 'zebras'] 

print rm_word_base
print rm_words

tag = '' 
full_vocabulary_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'  
vocab = 'vocabulary'
eightyK = False
if eightyK:
  full_vocabulary_file = '/z/lisaanne/pretrained_lm/yt_coco_surface_80k_vocab.txt'
  vocab = 'vocabulary80k'
beam1 = True
beam3 = False

gt_novel_json = '../../data/coco/coco/annotations/captions_split_set_%s_val_val_novel2014.json' %rm_word_base
gt_train_json = '../../data/coco/coco/annotations/captions_split_set_%s_val_val_train2014.json' %rm_word_base
set_novel = 'split_set_%s_val_val_novel' %rm_word_base
set_train = 'split_set_%s_val_val_train' %rm_word_base

#do sentence generation
beam1_json_path = retrieval_cache_home + 'val_val/all_ims/%s/beam1/generation_result.json' %(trained_model)
always_recompute = True
if beam1:
  if not os.path.exists(beam1_json_path) or always_recompute:
    experiment = {'type': 'generation'}
    retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name='val_val', vocab=vocab, precomputed_feats=None, feats_bool_in=feats_bool_in, precomputed_h5=feature_dir_h5, experiment=experiment)
  else:
    experiment = {'type': 'score_generation', 'json_file': beam1_json_path} 
    retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name='val_val', vocab=vocab, precomputed_feats=feature_dir, feats_bool_in=False, experiment=experiment)

if beam3:
  if not os.path.exists(retrieval_cache_home + '%s/all_ims/%s/beam3/generation_result.json' %(test_set, trained_model)):
    experiment = {'type': 'generation', 'beam_size': 3}
    retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name='val_val', vocab=vocab, precomputed_feats=feature_dir, feats_bool_in=False, experiment=experiment)

def eval_generation(generated_sentences, gt_file, test_set):
  generated_sentences_json = read_json(beam1_json_path)
  gt_json = read_json(gt_file)
  image_ids = list(np.unique([i['id'] for i in gt_json['images']]))  
  gen_novel = [g for g in generated_sentences_json if g['image_id'] in image_ids]
  tmp_json = 'gt_file_tmp.%s.%s.json' %(trained_model.split('/')[-1], rm_word_base)
  with open(tmp_json, 'w') as outfile:
    json.dump(gen_novel, outfile)
  experiment = {'type': 'score_generation', 'json_file': 'gt_file_tmp.json', 'read_file': True} 
  retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name=test_set, vocab=vocab, precomputed_feats=feature_dir, feats_bool_in=False, experiment=experiment)
  os.remove('gt_file_tmp.json')

print 'Scores for novel captions in val set:\n'
eval_generation(beam1_json_path, gt_novel_json, set_novel) 
print 'Scores for train captions in val set:\n'
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
print 'F1 score for beam1 is: %f' %f_score

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



