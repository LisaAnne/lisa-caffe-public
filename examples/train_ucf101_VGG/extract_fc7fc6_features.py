import numpy as np 
import re 
import pickle
import os

caffe_root = '../../' 
import sys 
sys.path.insert(0,caffe_root + 'python') 
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import glob

from multiprocessing import Pool
import random
#Things to change for flow: #vid_pattern #base_images #test_images
#parameters to vary

if len(sys.argv) > 3:
  is_flow = bool(int(sys.argv[3]))
else:
  is_flow = False
if len(sys.argv) > 4:
  split = int(sys.argv[4])
else:
  split = 1
deploy_net = 'VGG_16_layers_deploy.prototxt'

if is_flow:
  model = 'single_frame_all_layers_VGG_flow_split1_iter_70000.caffemodel'
  base_image_path = '/x/data/ucf101/flow_images_Georgia/'
else:
  model = 'single_frame_all_layers_VGG_RGB_split1_iter_6000.caffemodel'
  base_image_path = '/x/data/ucf101/frames/'
batch_size = 10

if is_flow:
  flow_or_RGB = 'flow'
else:
  flow_or_RGB = 'RGB'


#get list of flow images to run through VGG
test_list = open(('/x/data/ucf101/ucfTrainTestlist/%slist0%d.txt' %(train_or_test, split)), 'rb')

if len(sys.argv) > 1:
  start = int(sys.argv[1])
else: 
  start = 0
if len(sys.argv) > 2:
  end = int(sys.argv[2])
else: 
  end = len(test_list)
#test_list = open(('/x/data/ucf101/ucfTrainTestlist/trainlist0%d.txt' %split), 'rb')
action_hash = pickle.load(open('/x/data/ucf101/action_hash_ucf.p', 'rb'))

video_dict = {}
for line in test_list:
  video =  line.split('/')[1].split('.avi')[0]
  action = line.split('/')[0]
  if action == 'HandstandPushups':
    action = 'HandStandPushups'
  video_path = base_image_path + video
  video_dict[video] = {}
  video_dict[video]['frames'] = sorted(glob.glob('%s/*.jpg' %video_path))
  video_dict[video]['label'] = action_hash[action]

test_list.close()
end = min(end, len(video_dict.keys()))

#set up network

net = caffe.Net(deploy_net, model, caffe.TEST)

#set up transformer

shape = (128, 3, 224, 224)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_raw_scale('data', 255)
if is_flow:
  image_mean = [128, 128, 128]
  #transformer.set_is_flow('data', True)
else:
  image_mean = [103.939, 116.779, 128.68]
  #transformer.set_is_flow('data', False)
channel_mean = np.zeros((3,224,224))
for channel_index, mean_val in enumerate(image_mean):
  channel_mean[channel_index, ...] = mean_val
transformer.set_mean('data', channel_mean)
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_transpose('data', (2, 0, 1))


def image_processor(ins):
  input_im = ins[0]
  rand_x = ins[1]
  rand_y = ins[2]
  data_in = caffe.io.load_image(input_im)
  if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
    data_in = caffe.io.resize_image(data_in, (240,320))
  data_in = data_in[rand_y:rand_y+224, rand_x:rand_x+224,:]
  processed_image = transformer.preprocess('data',data_in)
  return processed_image

num_videos = end-start 

disp = 10
vid_correct = 0
vid_count = 0 

for video in video_dict.keys()[start:end]:
  frames = sorted(video_dict[video]['frames']) 
 
  if (vid_count + disp) % disp == 0: 
    if vid_count > 0: 
      accuracy = float(vid_correct)/float(vid_count)
      print '%s, %s: On video %d of %d; accuracy = %f\n' %(flow_or_RGB, train_or_test, vid_count, num_videos, accuracy) 
    else:
      print '%s, %s: On video %d of %d\n' %(flow_or_RGB, train_or_test, vid_count, num_videos) 
     
  label = video_dict[video]['label']
  
  save_name = '/y/lisaanne/ucf101/ucf101-%s_features_VGG/%s.p' %(flow_or_RGB, video)

  if not os.path.isfile(save_name):
        
    rand_x = int(random.random() * (320-224))
    rand_y = int(random.random() * (240-224))

    fc6 = np.zeros((0,4096))
    fc7 = np.zeros((0,4096))
    probs = np.zeros((0,101))

    for i in range(0,len(frames),batch_size):
      max_i = min(i + batch_size, len(frames))
      batch_frames = frames[i:max_i]
      ins = zip(batch_frames, [rand_x]*len(batch_frames), [rand_y]*len(batch_frames))
      data = pool.map(image_processor, ins)

      #data = []
      #for j in range(len(batch_frames)):
      #  data.append(image_processor((batch_frames[j], rand_x, rand_y)))

      net.blobs['data'].reshape((max_i-i),3,224,224)
      for j in range(max_i-i):
        net.blobs['data'].data[j,...] = data[j]
      out = net.forward()
      fc6 = np.concatenate((fc6, net.blobs['fc6'].data), axis = 0)
      fc7 = np.concatenate((fc7, net.blobs['fc7'].data), axis = 0)
      probs = np.concatenate((probs, out['probs']), axis = 0)
    net_out = {}
    net_out['video'] = video
    net_out['probs'] = probs 
    net_out['fc6'] = fc6
    net_out['label'] = label
    net_out['rand_x'] = rand_x
    net_out['rand_y'] = rand_y

    predicted_label = np.argmax(np.mean(probs,0))
    if predicted_label == label:
      vid_correct += 1 
 
    pickle.dump(net_out, open(save_name,'wb'))
  
  vid_count += 1

print "Final accuracy is %f." %(float(vid_correct)/float(vid_count))

