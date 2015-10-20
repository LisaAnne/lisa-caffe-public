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

#used to create a new coco image json
def create_coco_json(image_list_json, lexical_list, save_name):
  
  train_json = image_json()
  train_json.create_new_image_json()
  lexical_list = open(lexical_list).readlines()
  lexical_list = [l.strip() for l in lexical_list] 
  for item in lexical_list:
    train_json.add_new_label(item)
  
  image_list = read_json(image_list_json)
  
  train_json.add_new_dataset('coco', image_list)
  train_json.save_new_json(save_name)
  print 'Wrote train json to %s.' %save_name

def add_imagenet_images(current_json, objects):
  train_json = image_json()
  train_json.open_image_json(current_json)
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
  print 'Saved imagenet items to json %s.' %(current_json)


#create coco json

#original attributes (has 471 classes)
#lexical_list = 'utils_trainAttributes/lexicalList_parseCoco_JJ100_NN300_VB100.txt' #lexical list
#imageList_train = 'utils_trainAttributes/imageList_coco_train_parse_labels.json' #list of images and labels
#imageList_val_val = 'utils_trainAttributes/imageList_coco_test_parse_labels.json' #list of images and labels
#save_image_json_train = 'utils_trainAttributes/imageJson_train.json'
#save_image_json_val_val = 'utils_trainAttributes/imageJson_test.json'

#For expanded attribute set (has 715 classes)
lexical_list = 'utils_trainAttributes/lexicalList_parseCoco_JJ155_NN511_VB100.txt'
imageList_train = 'utils_trainAttributes/imageList_JJ155_NN511_VB100_coco_train.json'
imageList_val_val = 'utils_trainAttributes/imageList_JJ155_NN511_VB100_coco_val_val.json'
save_image_json_train = 'utils_trainAttributes/imageJson_JJ155_NN511_VB100_train.json'
save_image_json_val_val = 'utils_trainAttributes/imageJson_JJ155_NN511_VB100_val_val.json'

create_coco_json(imageList_train, lexical_list, save_image_json_train)
create_coco_json(imageList_val_val, lexical_list, save_image_json_val_val)

#Add imagenet objects
#objects = ['bottle', 'bus', 'couch', 'luggage', 'microwave', 'motorcycle', 'pizza', 'racket', 'suitcase', 'zebra']
#add_imagenet_images('utils_trainAttributes/imageJson_train.json', objects)

