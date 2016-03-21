import numpy as np
import sys
import glob
import os
import h5py
from multiprocessing import Pool
caffe_root = '../../'
sys.path.insert(0, '../../python/')
import caffe
sys.path.insert(0, '../')
from utils import caffe_utils
from utils.config import *
import pdb

def extract_features(model, model_weights, imagenet_ims=None, device=1, image_dim=227, feature_extract='probs', batch_size=10):
  caffe.set_mode_gpu()
  caffe.set_device(device)

  net_type = 'alex'
  if image_dim == 224:
    net_type = 'vgg'

  im_ids_train = open('../../data/coco/coco2014_cocoid.train.txt').readlines()
  im_ids_train = [int(im_id.strip()) for im_id in im_ids_train]
  im_ids_val = open('../../data/coco/coco2014_cocoid.val_val.txt').readlines()
  im_ids_val = [int(im_id.strip()) for im_id in im_ids_val]
  im_ids_test = open('../../data/coco/coco2014_cocoid.val_test.txt').readlines()
  im_ids_test = [int(im_id.strip()) for im_id in im_ids_test]

  train_ims = [coco_template %('train', 'train', im_id) for im_id in im_ids_train]
  val_ims = [coco_template %('val', 'val', im_id) for im_id in im_ids_val]
  test_ims = [coco_template %('val', 'val', im_id) for im_id in im_ids_test]

  sets = [train_ims, val_ims, test_ims]
  set_names = ['train', 'val_val', 'val_test']
  if imagenet_ims:
    #imagenet_ims = open('../captions_add_new_word/utils_trainAttributes/test_images_rebuttal.txt').readlines()
    imagenet_ims = open(imagenet_ims).readlines()
    test_imagenet = ['imagenet_images/' + i.strip() for i in imagenet_ims]
    sets.append(test_imagenet)
    set_names.append('imagenet_ims')


  save_h5 = model_weights.split('/')[-1]
  oversample_dim = True

  net = caffe.Net(model, model_weights, caffe.TEST)
  feature_size = net.blobs[feature_extract].data.shape[1]

  transformer = caffe_utils.build_transformer(image_dim)
  my_image_processor = lambda x: caffe_utils.image_processor(transformer, x, True)

  for s, set_name in zip(sets, set_names):
    all_ims = s
    features = np.zeros((len(all_ims), feature_size))
    for ix in range(0, len(all_ims), batch_size):
      sys.stdout.write('\rOn set %s. On image %d/%d.' %(set_name, ix, len(all_ims)))
      sys.stdout.flush()

      batch_end = min(ix+batch_size, len(all_ims))
      batch_frames = all_ims[ix:batch_end]
      data = []
      for b in batch_frames:
        data.extend(my_image_processor(b))
      net.blobs['data'].reshape(len(data),3,image_dim,image_dim)
      net.blobs['data'].data[...] = data
      out = net.forward()
      features_tmp = net.blobs[feature_extract].data
      features_av = np.array([np.mean(features_tmp[i:i+10], axis=0) for i in range(0, len(data), 10)])
      features[ix:ix+features_av.shape[0],:] = features_av

    h5_file = 'lexical_features/%s_feats.%s.%s.h5' %(net_type, save_h5, set_name)
    f = h5py.File(h5_file, "w")
    print "Printing to %s\n" %h5_file
    all_ims_short = [i.split('/')[-2] + '/' + i.split('/')[-1] for i in all_ims]
    assert len(all_ims_short) == len(features)
    dset = f.create_dataset("ims", data=all_ims_short)
    dset = f.create_dataset("features", data=features)
    f.close()
 









