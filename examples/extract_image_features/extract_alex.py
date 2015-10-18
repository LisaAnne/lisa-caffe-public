import numpy as np
import sys
import glob
import os
import h5py

caffe_root = '../../'
sys.path.insert(0, '../../python/')
import caffe
from multiprocessing import Pool
#coco images
image_path = '../../data/coco/coco/images/'
sys.path.insert(0, '../captions_add_new_word/')
#imagnet images
#image_path = '/y/lisaanne/imageData/imagenet/'
#sets = ['pizza', 'zebra', 'motorcycle']
feature_path = '/y/lisaanne/lexical_features/'
coco_template = '../../data/coco/coco/images/%s2014/COCO_%s2014_%012d.jpg'
im_ids_train = open('../../data/coco/coco2014_cocoid.no_caption_zebra_train.txt').readlines()
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

#sets = ['test2014','train2014']
#sets = ['train2014']
caffe.set_mode_gpu()
caffe.set_device(0)

#vgg weights
#model_file = '../../models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt'
#model_weights = '../../models/vgg/VGG_ILSVRC_16_layers.caffemodel'
#image_dim = 224
#model_file = '../captions_add_new_word/train_classifiers.vgg.deploy.prototxt'
#model_weights = '/x/lisaanne/coco_attribute/train_lexical_classifier/attributes_JJ100_NN300_VB100_imagenet_zebra.vgg_iter_50000'

#lexical weights
model_file = '../coco_attribute/mrnn_attributes_fc8-probs_deploy.prototxt'
#model_file = '../captions_add_new_word/train_classifiers_deploy.prototxt'
model_weights = '/x/lisaanne/coco_attribute/train_lexical_classifier/attributes_JJ100_NN300_VB100_eightClusters_cocoImages_iter_50000'
#model_weights = '/x/lisaanne/coco_attribute/train_lexical_classifier/attributes_JJ100_NN300_VB100_zebra_iter_50000'
save_h5 = model_weights.split('/')[-1]
image_dim = 227
oversample_dim = True
feature_extract = 'prob-attributes'
feature_size = 471

net = caffe.Net(model_file, model_weights + '.caffemodel', caffe.TEST)
shape = (128,3,image_dim,image_dim)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_raw_scale('data', 255)
image_mean = [103.939, 116.779, 128.68]
channel_mean = np.zeros((3,image_dim,image_dim))
for channel_index, mean_val in enumerate(image_mean):
  channel_mean[channel_index, ...] = mean_val
transformer.set_mean('data', channel_mean)
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_transpose('data', (2, 0, 1))

def image_processor(input_im):
  data_in = caffe.io.load_image(input_im)
  data_in = caffe.io.resize_image(data_in, (256,256))
  # take center crop
  if oversample_dim:
    oversampled_image = caffe.io.oversample([data_in], (image_dim, image_dim))
    processed_image = []
    for oi in oversampled_image:
      processed_image.append(transformer.preprocess('data', oi))
  else:
    shift = (256-image_dim)/2
    data_in = data_in[shift:shift+image_dim,shift:shift+image_dim,:]
    processed_image = transformer.preprocess('data',data_in)
  return processed_image

batch_size = 50 
for s, set_name in zip(sets, set_names):
  all_ims = s
  features = np.zeros((len(all_ims), feature_size))
  for ix in range(0, len(all_ims), batch_size):
    count_im = ix/batch_size
    print 'On set %s. On image %d/%d.' %(set_name, ix, len(all_ims))
    batch_end = min(count_im*batch_size+batch_size, len(all_ims))
    batch_frames = all_ims[count_im*batch_size:batch_end]
    data = []
    for b in batch_frames:
      if oversample_dim:
        data.extend(image_processor(b))
      else:
        data.append(image_processor(b))
    net.blobs['data'].reshape(len(data),3,image_dim,image_dim)
    net.blobs['data'].data[...] = data
    out = net.forward()
    features_tmp = net.blobs[feature_extract].data
    if oversample_dim:
      features_av = [np.mean(features_tmp[i:i+10], axis=0) for i in range(0, len(data), 10)]
      features_tmp = np.array(features_av)
    features[ix:ix+features_tmp.shape[0],:] = features_tmp
  h5_file = '/y/lisaanne/lexical_features/alex_feats.%s.%s.h5' %(save_h5, set_name)
  f = h5py.File(h5_file, "w")
  print "Printing to %s\n" %h5_file
  all_ims_short = [i.split('/')[-2] + '/' + i.split('/')[-1] for i in all_ims]
  assert len(all_ims_short) == len(features)
  dset = f.create_dataset("ims", data=all_ims_short)
  dset = f.create_dataset("features", data=features)
  f.close()
 








