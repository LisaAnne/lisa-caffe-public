import numpy as np
import json
import retrieval_experiment
import sys
import os
import argparse

#This script will run all the following tests:
	#(1) Sentence generation for beam1 and beam3
	#(2) F1-score
        #(3) Fill in the blank metrics
	#(4) Vocabulary comparison metrics


#example input: python -m pdb analyze_new_word.py augment_train_noZebra_vocab8749_iter_110000 /models/bvlc_reference_caffenet/deploy.prototxt /examples/coco_caption/lrcn_word_to_preds.deploy.prototxt augment_train_noZebra_ 1 1

home_dir = '../../'
retrieval_cache_home = '/x/lisaanne/retrieval_cache/'

def read_json(json_file):
  t_json = open(json_file).read()
  return json.loads(t_json) 

rm_word_base = 'zebra'
rm_words = ['zebra', 'zebras']
#rm_word_base = 'pizza'
#rm_words = ['pizza', 'pizzas']
rm_word_base = 'suitcase'
rm_words = ['luggage', 'luggages', 'suitcase', 'suitcases']

#rm_word_base = 'rm_eightCluster'
#rm_words = ['luggage', 'luggages', 'suitcase', 'suitcases', 'bottle', 'bottles', 'couch', 'couches', 'sofa', 'so    fas', 'microwave', 'microwaves', 'rackett', 'racket', 'raquet', 'rackets',  'bus', 'buses', 'busses', 'pizza', 'pizz    as', 'zebra', 'zebras'] 

trained_model = sys.argv[1]
trained_model = trained_model
image_model = sys.argv[2]
language_model = sys.argv[3]
if len(sys.argv) > 4:
  feature_dir_h5 = sys.argv[4]
  feats_bool_in = True
else: 
  feature_dir_h5 = None
  feats_bool_in = False
if len(sys.argv) > 5:
  feature_dir = sys.argv[4]
else: 
  feature_dir = None

tag = '' 
full_vocabulary_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'  
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
    retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name='val_val', vocab='vocabulary', precomputed_feats=None, feats_bool_in=feats_bool_in, precomputed_h5=feature_dir_h5, experiment=experiment)
  else:
    experiment = {'type': 'score_generation', 'json_file': beam1_json_path} 
    retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name='val_val', vocab='vocabulary', precomputed_feats=feature_dir, feats_bool_in=False, experiment=experiment)

if beam3:
  if not os.path.exists(retrieval_cache_home + '%s/all_ims/%s/beam3/generation_result.json' %(test_set, trained_model)):
    experiment = {'type': 'generation', 'beam_size': 3}
    retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name='val_val', vocab='vocabulary', precomputed_feats=feature_dir, feats_bool_in=False, experiment=experiment)

def eval_generation(generated_sentences, gt_file, test_set):
  generated_sentences_json = read_json(beam1_json_path)
  gt_json = read_json(gt_file)
  image_ids = list(np.unique([i['id'] for i in gt_json['images']]))  
  gen_novel = [g for g in generated_sentences_json if g['image_id'] in image_ids]
  with open('gt_file_tmp.json', 'w') as outfile:
    json.dump(gen_novel, outfile)
  experiment = {'type': 'score_generation', 'json_file': 'gt_file_tmp.json', 'read_file': True} 
  retrieval_experiment.main(model_name=trained_model, image_net=image_model, LM_net=language_model, dataset_name=test_set, vocab='vocabulary', precomputed_feats=feature_dir, feats_bool_in=False, experiment=experiment)
  os.remove('gt_file_tmp.json')

print 'Scores for novel captions in val set:\n'
eval_generation(beam1_json_path, gt_novel_json, set_novel) 
print 'Scores for train captions in val set:\n'
eval_generation(beam1_json_path, gt_train_json, set_train) 


##compute f1-score for missing words using generated captions
#def filter_sentence(sentence):
#  return sentence.replace('.','').replace(',','').replace("'",'').lower().split()
#
#def match_words(rm_words, words):
#  if not rm_words:
#    return False #if rm_words is none want to return False.
#  list_matches = [False]*len(rm_words)
#  for x, rm_w in enumerate(rm_words):
#    list_matches[x] = any([w == rm_w for w in words])
#  return any(list_matches)
#
#def compute_f1_score(beam):
#  gen_novel_captions = read_json('../../retrieval_cache/%s/all_ims/%s/%s/generation_result.json' %(test_sets[1], trained_model, beam))
#  gen_train_captions = read_json('../../retrieval_cache/%s/all_ims/%s/%s/generation_result.json' %(test_sets[2], trained_model, beam))
#  #true positive are sentences that contain match words and should
#  tp = sum(map(int, [match_words(rm_words, filter_sentence(c['caption'])) for c in gen_novel_captions])) 
#  #false positive are sentences that contain match words and should not
#  fp = sum(map(int, [match_words(rm_words, filter_sentence(c['caption'])) for c in gen_train_captions])) 
#  #false positive are sentences that do not contain match words and should
#  fn = sum(map(int, [not match_words(rm_words, filter_sentence(c['caption'])) for c in gen_novel_captions]))
# 
#  #precision = tp/(tp+fp)
#  precision = float(tp)/(tp+fp) 
#  #recall = tp/(tp+fn)
#  recall = float(tp)/(tp+fn)
#  #f1 = 2* (precision*recall)/(precision+recall)
#  return 2*(precision*recall)/(precision+recall)
#
#if beam1:
#  f_score_beam1 = compute_f1_score('beam1')
#  print 'F1 score for beam1 is: %f' %f_score_beam1
#if beam3: 
#  f_score_beam3 = compute_f1_score('beam3')
#  print 'F1 score for beam3 is: %f' %f_score_beam3

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



