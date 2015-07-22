import numpy as np 
import re 
import pickle
import os

caffe_root = '../../../' 
import sys 
sys.path.insert(0,caffe_root + 'python') 
import caffe
caffe.set_mode_gpu()
caffe.set_device(1)
import glob
from multiprocessing import Pool

import random
pool_size = 24
pool = Pool(processes=pool_size)
#Things to change for flow: #vid_pattern #base_images #test_images

base_folder = '/home/lisa/caffe-LSTM-video/data/coco/coco/images/'
MODEL_FILE = 'VGG_ILSVRC_16_layers_deploy.prototxt'
PRETRAINED = 'VGG_ILSVRC_16_layers.caffemodel'

image_folder = sys.argv[1]

images = glob.glob(('%s/%s/*.jpg' %(base_folder, image_folder)))
if len(sys.argv) > 2:
  start = int(sys.argv[2])
  end = int(sys.argv[3])
else:
  start = 0
  end = len(images)


net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)

shape = (128, 3, 224, 224)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_raw_scale('data', 255)
image_mean = [103.939, 116.779, 128.68]
channel_mean = np.zeros((3,224,224))
for channel_index, mean_val in enumerate(image_mean):
  channel_mean[channel_index, ...] = mean_val
transformer.set_mean('data', channel_mean)
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_transpose('data', (2, 0, 1))

def image_processor(input_im):
  data_in = caffe.io.load_image(input_im)
  if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
    data_in = caffe.io.resize_image(data_in, (256,256))
  rand_x = int(random.random() * (256-224))
  rand_y = int(random.random() * (256-224))
  data_in = data_in[rand_x:rand_x+224, rand_y:rand_y+224,:]
  processed_image = transformer.preprocess('data',data_in)
  return processed_image

num_images = end-start 

disp = 5
batch_size = 128

for image_count, image in enumerate(images[start:end:batch_size]):
 
  if image_count % disp == 0: 
    print 'On image %d of %d\n' %(image_count*batch_size, num_images) 

  trunc_batch_size = min(image_count*batch_size+batch_size, len(images))-image_count*batch_size 
  batch_frames = images[image_count*batch_size:image_count*batch_size+trunc_batch_size] 
  data = []
#  for j in range(len(batch_frames)):
#    data.append(image_processor(batch_frames[j]))
  data = pool.map(image_processor, batch_frames)

  net.blobs['data'].reshape((trunc_batch_size),3,224,224)
  for j in range(trunc_batch_size):
    net.blobs['data'].data[j,...] = data[j]
  out = net.forward()
  fc7 = net.blobs['fc7'].data

  for i, feat in enumerate(fc7):
    im_name = batch_frames[i]
    save_name = '/y/lisaanne/image_captioning/coco_features/%s_vgg_fc7Feat.p' %(im_name.split('/')[-1].split('.')[0])
    net_out = {}
    net_out['fc7'] = feat
    net_out['im_name'] = im_name
   
    pickle.dump(net_out, open(save_name,'wb'))
  



