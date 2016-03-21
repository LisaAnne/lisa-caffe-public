import sys
sys.path.insert(0, '../../python/')
import caffe
import numpy as np
import pdb

def build_transformer(image_dim):
  transformer = caffe.io.Transformer({'data': (1, 3, image_dim, image_dim)})
  transformer.set_raw_scale('data', 255)
  image_mean = [103.939, 116.779, 128.68]
  channel_mean = np.zeros((3,image_dim,image_dim))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  return transformer

def image_processor(transformer, input_im, oversample=False):
  data_in = caffe.io.load_image(input_im)
  data_in = caffe.io.resize_image(data_in, (256,256))
  image_dim = transformer.mean['data'].shape[-1]
  # take center crop
  if oversample:
    oversampled_image = caffe.io.oversample([data_in], (image_dim, image_dim))
    processed_image = []
    for oi in oversampled_image:
      processed_image.append(transformer.preprocess('data', oi))
  else:
    shift = (256-image_dim)/2
    data_in = data_in[shift:shift+image_dim,shift:shift+image_dim,:]
    processed_image = [transformer.preprocess('data',data_in)]
  return processed_image

