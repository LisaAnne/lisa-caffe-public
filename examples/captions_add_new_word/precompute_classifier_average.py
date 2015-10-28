import h5py
import json
import hickle as hkl
import numpy as np
import time

#load precomputed trained features
train_features_h5 = '/y/lisaanne/lexical_features/alex_feats.attributes_JJ100_NN300_VB100_eightClusters_cocoImages_iter_50000.train.h5'
train_features_h5 = '/y/lisaanne/lexical_features/alex_feats.attributes_JJ100_NN300_VB100_eightClusters_cocoImages_iter_50000.val_val.h5'
#train_features_h5 = '/y/lisaanne/lexical_features/alex_feats.attributes_JJ100_NN300_VB100_eightClusters_imagenetImages_1026_iter_25000.val_val.h5'

#image_labels
train_labels_json = 'utils_trainAttributes/imageJson_train.json'
train_labels_json = 'utils_trainAttributes/imageJson_test.json'

#label list
labels = open('utils_trainAttributes/lexicalList_parseCoco_JJ100_NN300_VB100.txt').readlines()
labels = [l.strip() for l in labels]

def read_json(json_file):
  t_json = open(json_file).read()
  return json.loads(t_json)

train_labels = read_json(train_labels_json)

precomputed = h5py.File(train_features_h5, 'r')
precomputed_ims = precomputed['ims']
precomputed_feats = precomputed['features']
precomputed_features_dict = {}
for i in range(len(precomputed_ims)):
  precomputed_features_dict[precomputed_ims[i]] = precomputed_feats[i]

#will have positive images for each label
labeled_ims = {}
label_hash = {}
for ix, l in enumerate(labels):
  labeled_ims[l] = []
  label_hash[l] = ix

for key in train_labels['images']['coco']:
  positive_labels = train_labels['images']['coco'][key]['positive_label']
  for p in positive_labels:
    labeled_ims[p].append(key)


def average_positive_classifiers():
  average_weights = {}
  
  for label in labeled_ims.keys():
    average_weight = 0
    label_idx = label_hash[label]
    for p in labeled_ims[label]:
      average_weight +=  precomputed_features_dict[p][label_idx] 
    average_weight /= len(labeled_ims[label])
    average_weights[label] = average_weight
  
  precomputed.close()
  
  hkl.dump(average_weights, 'utils_trainAttributes/average_weights_zebra.hkl')

def average_all_positive_classifiers():
  average_weights = {}
  
  for label in labeled_ims.keys():
    t = time.time()
    print label
    average_weight = np.zeros((471,471))
    label_idx = label_hash[label]
    for ix, other_label in enumerate(labeled_ims.keys()):
      average_weight_o = np.zeros((471,))     
      count_paths = 0
      unique_ims = [i for i in labeled_ims[other_label] if i not in labeled_ims[label]] 
      if label == other_label:
        unique_ims = labeled_ims[label]
      for p in unique_ims:
        average_weight_o +=  precomputed_features_dict[p] 
        count_paths += 1
      if count_paths > 0:
        average_weight_o /= count_paths
      average_weight[:,ix] = average_weight_o
    average_weights[label] = average_weight
    print "Time to compute one labels is %f\n" %(t-time.time())
    #hkl.dump(average_weights, 'utils_trainAttributes/average_weights_zebra.hkl')
  
  precomputed.close()
  
average_positive_classifiers()

