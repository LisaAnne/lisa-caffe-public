#If you are not getting ~71% when you evaluate the LRCN code for RGB images, you can use this script to help figure out what the issue is. 

import argparse
import evaluate_lstm
import glob
import h5py
import os
import numpy as np
import matplotlib.image as mpimg
import sys
caffe_root = '../../'
sys.path.insert(0, '../../python/')
import caffe
caffe.set_mode_gpu()
caffe.set_device(1)
import pickle as pkl

parser = argparse.ArgumentParser()
#Model to evaluate
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--im_path", type=str, default='frames')
deploy_im = 'deploy_lstm_im.prototxt'
deploy_lstm ='deploy_lstm_lstm_check.prototxt'
tol = 0.00001
init_debug = False 

args = parser.parse_args()

if not args.model:
  raise Exception("Must input trained model for evaluation") 

videos = [('v_JugglingBalls_g01_c02', 45),
          ('v_MilitaryParade_g05_c03', 52)] 
#          ('v_Skiing_g05_c03', 80),
#          ('v_Diving_g06_c03', 25), 
#          ('v_BrushingTeeth_g05_c05', 19),
#          ('v_PlayingDhol_g04_c06', 60),
#          ('v_CuttingInKitchen_g06_c01', 24),
#          ('JumpRope/v_JumpRope_g06_c04', 47),
#          ( 'Lunges/v_Lunges_g03_c02', 51),
#          ('v_ThrowDiscus_g03_c02', 92)]

def read_pkl(open_file):
  return pkl.load(open(open_file, 'r')) 
 
def save_pkl(dump_dict, save_file):
  pkl.dump(dump_dict, open(save_file, 'w')) 

def check_values(check1, check2):
  return abs(check1 - check2 <= tol), abs(check1 - check2)

if not init_debug:
  check_dict = read_pkl('check_dict.p')

#extract features
video_dict = {}
if init_debug: check_dict = {}
for video, label in videos:
  video_dict[video] = {}
  video_full_path = args.im_path + '/' + video  
  video_dict[video]['label'] = label
  video_dict[video]['frames'] = sorted(glob.glob('%s/*jpg' %video_full_path))
  if init_debug: check_dict[video] = {}

net = caffe.Net(deploy_im, args.model, caffe.TEST) 
print "Test: Checking that all image weights are equivalent..."
if init_debug: check_dict['image_weights'] = {}
for param in net.params.keys():
  if init_debug: check_dict['image_weights'][param] = {}
  for ip, param_item in enumerate(net.params[param]):
    if init_debug: check_dict['image_weights'][param][ip] = {}
    param_weights = param_item.data
    min_weights = np.min(param_weights)
    max_weights = np.min(param_weights)
    if init_debug: check_dict['image_weights'][param][ip]['min'] = min_weights
    if init_debug: check_dict['image_weights'][param][ip]['max'] = max_weights
    check, value = check_values(check_dict['image_weights'][param][ip]['min'], min_weights)
    if not check:
      raise Exception("Min image weights for layer %s (%ip) deviate by: %f.\n" %(param, ip, value)) 
    check, value = check_values(check_dict['image_weights'][param][ip]['max'], max_weights)
    if not check:
      raise Exception("Max image weights for layer %s (%ip) deviate by: %f.\n" %(param, ip, value)) 

print "Pass:  All image weights equivalent.\n"

print "Test: Checking that frames extracted correclty..."
for video in video_dict.keys():
  if init_debug: check_dict[video]['frame_extraction'] = {}
  if init_debug: check_dict[video]['frame_extraction'] = {}
  len_video = len(video_dict[video]['frames'])
  if init_debug: check_dict[video]['frame_extraction']['length'] = len_video
  check, value = check_values(check_dict[video]['frame_extraction']['length'], len_video)
  if not check:
    raise Exception("Number frames in video %s deviates by: %f.\n" %(video, value))
  if not check:
    raise Exception("Min pixel value for video %s frame %d deviates by: %f.\n" %(video, frame, value))
  for frame in [0, 15, len_video-1]:
    if init_debug: check_dict[video]['frame_extraction'][frame] = {} 
    im = mpimg.imread(video_dict[video]['frames'][frame])
    if init_debug: check_dict[video]['frame_extraction'][frame]['min'] = np.min(im)
    if init_debug: check_dict[video]['frame_extraction'][frame]['max'] = np.max(im)
    if init_debug: check_dict[video]['frame_extraction'][frame]['mean'] = np.mean(im)
    if init_debug: check_dict[video]['frame_extraction'][frame]['shape0'] = im.shape[0]
    if init_debug: check_dict[video]['frame_extraction'][frame]['shape1'] = im.shape[1]
    check, value = check_values(check_dict[video]['frame_extraction'][frame]['min'], np.min(im))
    if not check:
      raise Exception("Min pixel value for video %s frame %d deviates by: %f.\n" %(video, frame, value))
    check, value = check_values(check_dict[video]['frame_extraction'][frame]['max'], np.max(im))
    if not check:
      raise Exception("Max pixel value for video %s frame %d deviates by: %f.\n" %(video, frame, value))
    check, value = check_values(check_dict[video]['frame_extraction'][frame]['mean'], np.mean(im))
    if not check:
      raise Exception("Mean pixel value for video %s frame %d deviates by: %f.\n" %(video, frame, value))
    check, value = check_values(check_dict[video]['frame_extraction'][frame]['shape0'], im.shape[0])
    if not check:
      raise Exception("Height for video %s frame %d deviates by: %f.\n" %(video, frame, value))
    check, value = check_values(check_dict[video]['frame_extraction'][frame]['shape1'], im.shape[1])
    if not check:
      raise Exception("Width for video %s frame %d deviates by: %f.\n" %(video, frame, value))

print "Pass:  Image frames extracted correctly.\n"

transformer = evaluate_lstm.create_transformer(net, 227, False) 
print "Test: Checking that all forward blobs are equivalent..."
for video in video_dict.keys():
  if init_debug: check_dict[video]['image_blobs'] = {}
  
  frames = video_dict[video]['frames']
  transformed_data = []
  for frame in sorted(frames[:16]):
    transformed_data.append(evaluate_lstm.image_processor(transformer, frame))
  net.blobs['data'].reshape(16, 3, 227, 227)
  net.forward()
  for blob in net.blobs.keys():
    if init_debug: check_dict[video]['image_blobs'][blob] = {}
    frame_blob = net.blobs[blob].data[0,...]
    if init_debug: check_dict[video]['image_blobs'][blob]['min'] = np.min(frame_blob)
    if init_debug: check_dict[video]['image_blobs'][blob]['max'] = np.max(frame_blob)
    check, value = check_values(check_dict[video]['image_blobs'][blob]['min'], np.min(frame_blob))
    if not check:
      raise Exception("Forward blob (%s) min value for video %s (frame 1) deviates by: %f.\n" %(blob, video, value))
    check, value = check_values(check_dict[video]['image_blobs'][blob]['max'], np.max(frame_blob))
    if not check:
      raise Exception("Forward blob (%s) max value for video %s (frame 1) deviates by: %f.\n" %(blob, video, value))
    
  video_dict[video]['features'] = net.blobs['fc6'].data

print "Pass:  Image blobs extracted correctly.\n"

del net
net = caffe.Net(deploy_lstm, args.model, caffe.TEST) 

#check LSTM weights
print "Test: Checking that all lstm weights are equivalent..."
if init_debug: check_dict['lstm_weights'] = {}
for param in net.params.keys():
  if init_debug: check_dict['lstm_weights'][param] = {}
  for ip, param_item in enumerate(net.params[param]):
    if init_debug: check_dict['lstm_weights'][param][ip] = {}
    param_weights = param_item.data
    if init_debug: check_dict['lstm_weights'][param][ip]['min'] = np.min(param_weights)
    if init_debug: check_dict['lstm_weights'][param][ip]['max'] = np.max(param_weights)
    check, value = check_values(check_dict['lstm_weights'][param][ip]['min'], np.min(param_weights))
    if not check:
      raise Exception("LSTM weights for blob %s (%d) min value deviates by: %f.\n" %(param, ip, value))
    check, value = check_values(check_dict['lstm_weights'][param][ip]['max'], np.max(param_weights))
    if not check:
      raise Exception("LSTM weights for blob %s (%d) max value deviates by: %f.\n" %(param, ip, value))

print "Pass:  All lstm weights equivalent.\n"

clip_markers = np.ones((16, 1, 1, 1))
clip_markers[0] = 0
net.blobs['data'].reshape(16, 4096, 1, 1)
net.blobs['clip_markers'].reshape(16, 1, 1, 1)
net.blobs['clip_markers'].data[...] = clip_markers
for video in video_dict.keys():
  if init_debug: check_dict[video]['lstm_blobs'] = {}
  data_in = np.zeros((16, 4096, 1, 1))
  data_in[:,:,0,0] = video_dict[video]['features']  
  net.blobs['data'].data[...] = data_in
  out = net.forward()
  for blob in net.blobs.keys():
    if init_debug: check_dict[video]['lstm_blobs'][blob] = {}
    sequence_blob = net.blobs[blob].data[...]
    print "For video %s (frame 1) and blob %s, min value is %f and max value is %f.\n" %(video, blob, np.min(sequence_blob), np.max(sequence_blob)) 
    if init_debug: check_dict[video]['lstm_blobs'][blob]['min'] = np.min(sequence_blob)
    if init_debug: check_dict[video]['lstm_blobs'][blob]['max'] = np.max(sequence_blob)
    check, value = check_values(check_dict[video]['lstm_blobs'][blob]['min'], np.min(sequence_blob))
    if not check:
      raise Exception("Forward blob (%s) min value for video %s deviates by: %f.\n" %(blob, video, value))
    check, value = check_values(check_dict[video]['lstm_blobs'][blob]['max'], np.max(sequence_blob))
    if not check:
      raise Exception("Forward blob (%s) max value for video %s deviates by: %f.\n" %(blob, video, value))
  
print "Done with debugging check -- passed all tests"
if init_debug:
  save_pkl(check_dict, 'check_dict.p') 
