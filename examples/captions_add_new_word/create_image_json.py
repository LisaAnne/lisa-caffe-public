#This file will create a json file which includes labeled images to pretrain the lexical layer.  The json file will have the following form:
#	json_file['images']['dataset']['path']['label index']
#       json_file['labels']['label name']['label index']
#dataset is something like "imagenet", "coco", etc.  Should specify the exact path in python data layer to be machine specific

import json
import os
import sys
import pickle as pkl
from init import *
import glob

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

class image_json(object):

  def create_new_image_json(self):
    #create a completely new image json
    image_json = {}
    image_json['images'] = {}
    image_json['labels'] = []
    self.image_json = image_json
  
  def open_image_json(self, saved_image_json):
    self.image_json = read_json(saved_image_json)
    self.json_save_name = saved_image_json

  def add_new_dataset(self, dataset, images):
    #dataset is name of dataset to be added
    #images is a dict with the following format: [image_path]['positive_label'] and [image_path]['negative_label']
 
    image_json = self.image_json
    if not dataset in image_json['images'].keys():
      image_json['images'][dataset] = {}
    
    for image in images.keys():
      image_json['images'][dataset][image] = {}
      image_json['images'][dataset][image]['positive_label'] = images[image]['positive_label']
      image_json['images'][dataset][image]['negative_label'] = images[image]['negative_label']
  
    self.image_json = image_json

  def add_new_images(self):
    raise Exception('Not yet implemented')

  def add_new_label(self, label):
    #add new label to list of labels
    self.image_json['labels'].append(label) 

  def save_updated_json(self):
    os.remove(self.json_save_name)
    with open(self.json_save_name, 'w') as outfile:
      json.dump(self.image_json, outfile) 

  def save_new_json(self, save_new_json_name):
    with open(save_new_json_name, 'w') as outfile:
      json.dump(self.image_json, outfile) 

train_json = image_json()
train_json.open_image_json('utils_trainAttributes/imageJson_train.json')
objects = ['bottle', 'bus', 'couch', 'luggage', 'microwave', 'motorcycle', 'pizza', 'racket', 'suitcase']

for o in objects:
  new_object_dict = {}
  object_paths = glob.glob('%s%s/*.JPEG' %(imagenet_root, o))
  for op in object_paths:
    op_partial = '/'.join(op.split('/')[-2:])
    new_object_dict[op_partial] = {}
    new_object_dict[op_partial]['positive_label'] = [o]
    new_object_dict[op_partial]['negative_label'] = []
  train_json.add_new_dataset('imagenet', new_object_dict)

train_json.save_updated_json()

#To create a json with coco images and zebra images from imagenet
#image_list_coco_train_json = 'utils_trainAttributes/imageList_coco_train_parse_labels.json'
#image_list_imagenet_zebra_json = 'utils_trainAttributes/imageList_imagenet_train_zebra.json'
#
#train_json = image_json()
#train_json.create_new_image_json()
#attributes = pkl.load(open('../coco_attribute/attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))
#for attribute in attributes:
#  train_json.add_new_label(attribute)
#
#image_list_coco_train = read_json(image_list_coco_train_json)
#image_list_imagenet_zebra = read_json(image_list_imagenet_zebra_json)
#
#train_json.add_new_dataset('coco', image_list_coco_train)
#train_json.add_new_dataset('imagenet', image_list_imagenet_zebra)
#train_json.save_new_json('utils_trainAttributes/imageJson_train.json')

#image_list_coco_test_json = 'utils_trainAttributes/imageList_coco_test_parse_labels.json'
#
#train_json = image_json()
#train_json.create_new_image_json()
#attributes = pkl.load(open('../coco_attribute/attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))
#for attribute in attributes:
#  train_json.add_new_label(attribute)
#
#image_list_coco_test = read_json(image_list_coco_test_json)
#
#train_json.add_new_dataset('coco', image_list_coco_test)
#train_json.save_new_json('utils_trainAttributes/imageJson_test.json')
