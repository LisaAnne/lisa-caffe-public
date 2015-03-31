#!/mnt/y/lisaanne/anaconda/bin/python

import numpy as np
import h5py
import pickle
import sys
import math
import os.path
import scipy.io

def normalize_joint_x_y(joints, num_joints=15):
  #joints stored as 2x15xn array
  #normalize coordinates as: (p-p_min)/(p_max-p_min) where 'p' is the coordinate
#  min_all = np.min(joints, axis=1)
#  max_all = np.max(joints, axis=1)
#  joints -= np.tile(min_all,[num_joints,1,1]).swapaxes(0,1)
#  joints = np.divide(joints, np.tile(max_all-min_all,[num_joints,1,1]).swapaxes(0,1))
#  return joints*2 - 1
  
  #place a single joint at zero
  joints[1,:,:] = np.max(joints[1,:,:], axis = 0) - joints[1,:,:]
  skeleton_zero = joints[:,1,:]
  joints -= np.tile(skeleton_zero,[num_joints,1,1]).swapaxes(0,1)
 
  min_all_x = np.min(joints[0,:,:])
  max_all_x = np.max(joints[0,:,:])
  min_all_y = np.min(joints[1,:,:])
  max_all_y = np.max(joints[1,:,:])
  diff_y = max_all_y - min_all_y
  diff_x = max_all_x - min_all_x
  max_diff = max(diff_y, diff_x)

  joints /= max_diff; #np.divide(joints, normalize)
  return joints*2 

#  min_all_x = np.min(joints[0,:,:], axis=0)
#  max_all_x = np.max(joints[0,:,:], axis=0)
#  min_all_y = np.min(joints[1,:,:], axis=0)
#  max_all_y = np.max(joints[1,:,:], axis=0)
#  diff_y = max_all_y - min_all_y
#  diff_x = max_all_x - min_all_x
#  normalize = np.tile(np.maximum(diff_y, diff_x),[num_joints,2,1]).swapaxes(0,1)
#  joints = np.divide(joints, normalize)
#  return joints*2 

def normalize_wrt_allTimeSteps(joints):
  min_joints = np.min(joints)
  max_joints = np.max(joints)
  joints -= min_joints
  joints /= (max_joints-min_joints)
  return joints - 0.5

def det_joints_diff(joints, num_joints=15):
  #determine difference between current and previous sets of joints
  prev_joints = np.roll(joints, 1, axis=2)
  prev_joints[:,:,0] = joints[:,:,0]
  joints_diff = joints-prev_joints
  #range for all joints
  #range_x = np.max(joints_diff[0,:,:], axis=1) - np.min(joints_diff[0,:,:], axis = 1) 
  #max_x_range = np.max(range_x)  #max range
  #range_y = np.max(joints_diff[1,:,:], axis=1) - np.min(joints_diff[1,:,:], axis = 1) 
  #max_y_range = np.max(range_y)  #max range
  #once max ranges determine, normalize

  #diff 32
  #max_x_range = np.max(joints_diff[0,:,:])
  #max_y_range = np.max(joints_diff[1,:,:])
  #joints_diff[0,:,:] /= max_x_range
  #joints_diff[1,:,:] /= max_y_range

  #diffMaxHW diff  
  #width = np.max(joints[0,:,:]) - np.min(joints[0,:,:])
  #height = np.max(joints[1,:,:]) - np.min(joints[1,:,:])
  #norm = max(width, height)
  #joints_diff /= norm

  #raw joints
  #do nothing additional

  return joints_diff

#  prev_joints = np.roll(joints, 1, axis=2)
#  prev_joints[:,:,0] = joints[:,:,0]
#  print prev_joints[:,1,:]
#  print joints[:,1,:]
#  joints_diff = joints-prev_joints
#  return joints_diff

#  prev_joints = np.roll(joints, 1, axis=2)
#  prev_joints[:,:,0] = joints[:,:,0]
#  print prev_joints[:,1,:]
#  print joints[:,1,:]
#  joints_diff = joints-prev_joints
#  return joints_diff

def traj_feat_vec(joints, num_joints=15):
  #traj features from Cordelia's paper require:
  #	1. Joints in relation to center  (num_joints*2 values)
  #	2. Translation of normalized joint  (num_joints*2 values)
  #	3. Direction of translational vector (num_joints values)
  joints = normalize_joint_x_y(joints, num_joints)  
  joints_diff = det_joints_diff(joints, num_joints)
  translation_vector = np.arctan(joints_diff[1,:,:]/joints_diff[0,:,:])
  translation_vector = np.nan_to_num(translation_vector)
  #Reshape vectors
  joints = joints.T.reshape((joints.shape[2],num_joints*2)) 
  joints_diff = joints_diff.T.reshape((joints_diff.shape[2],num_joints*2)) 
  translation_vector = translation_vector.T

  #concatenate vectors
  joints = np.concatenate((joints,joints_diff),axis=1) 
  joints = np.concatenate((joints,translation_vector),axis=1) 
  num_feat = joints.shape[1]

  #return joints as well as num_feat
  return joints 

def subsample_feat(joints, subsample):
  return joints[0::3,:]

def mirror_pose(joints):
 #assume joints passed in as 2x15xn array
 mid_all = (np.max(joints[0,:,:]) + np.min(joints[0,:,:]))/2
 joints[0,:,:] = mid_all-1*(joints[0,:,:]-mid_all)
 return joints

def add_joints_to_dict(joints_np,label_hdf, pad_frames, num_frames_txt, h5_dict, train_or_test='train',num_feat=30):
  #default assume that both train and test set treated the same
  num_frames = joints_np.shape[0]
  joints = np.zeros((pad_frames, num_feat, 1 ,1))
  joints[0:joints_np.shape[0],:,0,0] = joints_np
  h5_dict['joints'] = np.concatenate((h5_dict['joints'],joints), axis=0)
  label_add = np.ones((pad_frames,1,1,1))*label_hdf
  h5_dict['label'] = np.concatenate((h5_dict['label'],label_add), axis=0)
  cm = np.ones((pad_frames,1,1,1))
  cm[0,:,:,:] = 0
  h5_dict['clip_markers'] = np.concatenate((h5_dict['clip_markers'],cm), axis=0)
  wl = np.zeros((pad_frames,1,1,1))
  if train_or_test == 'train':
    wl[num_frames/2:num_frames-1,:,:,:] = 1 
  elif train_or_test == 'test':
    wl[num_frames/2:num_frames-1,:,:,:] = 1 
  h5_dict['weight_loss'] = np.concatenate((h5_dict['weight_loss'],wl),axis=0)
  return h5_dict
 
def add_joints_to_dict_reshape(joints_np,label_hdf, pad_frames, num_frames_txt, h5_dict, num_feat=30):
  joints_np = joints_np.T.reshape((joints_np.shape[2],num_feat)) 
  joints = np.zeros((pad_frames, num_feat, 1 ,1))
  joints[0:joints_np.shape[0],:,0,0] = joints_np
  h5_dict['joints'] = np.concatenate((h5_dict['joints'],joints), axis=0)
  label_add = np.ones((pad_frames,1,1,1))*label_hdf
  h5_dict['label'] = np.concatenate((h5_dict['label'],label_add), axis=0)
  cm = np.ones((pad_frames,1,1,1))
  cm[0,:,:,:] = 0
  h5_dict['clip_markers'] = np.concatenate((h5_dict['clip_markers'],cm), axis=0)
  wl = np.zeros((pad_frames,1,1,1))
  wl[num_frames_txt/2:num_frames_txt,:,:,:] = 1
  h5_dict['weight_loss'] = np.concatenate((h5_dict['weight_loss'],wl),axis=0)
  return h5_dict


def make_frame_major(h5_dict, batch_size, buffer_size, pad_frames):
  num_frames = batch_size/buffer_size;
  h5_dict_fm = {}
  for key in h5_dict.keys(): 
    final_len = len(h5_dict[key])
    total_vid = final_len/pad_frames
    batches_per_vid = pad_frames/num_frames
    #num batches is a bit more complex...
    num_batches = int(math.ceil(float(total_vid)/buffer_size)*batches_per_vid)
    pad_len = num_batches*batch_size
    #roll over is weird if you just try to use matrices unfortunately :(
    h5_dict_fm[key] = np.zeros((pad_len,h5_dict[key].shape[1],1,1))

    track_frames = 0
    true_vid_list = []
    for batch in range(0,num_batches):
      for vid in range(0,buffer_size):
        for frame in range(0,num_frames):
          true_vid = (batch/batches_per_vid)*buffer_size + vid
          true_frames = (batch % batches_per_vid) * num_frames + frame
          cm_id = true_vid*pad_frames + true_frames
          fm_id = batch*batch_size + frame*buffer_size + vid
          true_vid_list.append(true_vid)
          if true_vid < total_vid:  #need to check this bc padding means we will try to access frames that are not present in h5_dict
            h5_dict_fm[key][fm_id,:,:,:] = h5_dict[key][cm_id,:,:,:]

          track_frames += 1

  return h5_dict_fm

def write_h5_file(h5_dict, h5_name):
  f = h5py.File(h5_name)
  for key in h5_dict.keys():
    f.create_dataset(key, data=h5_dict[key])
  f.close() 

def process_skeleton_angles(joints):
# could make this cleaner and more general for other datasets...

# Given a list of joint indices that is 2x15xN, determineJointAngles will rewrite this in a form where all joints are given as angle between joint and parent joint.

#Belly and neck define reference axis.  First feature will be angle between reference axis and y-axis measured clockwise.  Second feature will be rotation from previous frame (measured clockwise).  Place belly at (0,0)

# face, right shoulder, left shoulder, right hip, left hip are all defined by reference to belly/neck reference axis.  Include change in angle in reference to belly/neck line

# first frame make the angle (in radians 0); all frames afterwards are difference in angle from previous frame.

# right shoulder -> right elbow; left shoulder -> left elbow; left elbow -> left wrist; right elbow -> right wrist; right hip -> right knee; left hip -> left knee; riht knee -> right ankle; left knee -> left ankle

#0: neck
#1: belly
#2: face
#3: right shoulder
#4: left  shoulder
#5: right hip
#6: left  hip
#7: right elbow
#8: left elbow
#9: right knee
#10: left knee
#11: right wrist
#12: left wrist
#13: right ankle
#14: left ankle

#If I have joints a,b,c and want to find the angle made by a,b,c, I need to do the follosing:  x = (a-b), y = (c-b); alpha = acos(x*y/(norm(x)*norm(y)))
#Need to include angles as well as length of joints

  #change all x, y coordinates so belly is at (0,0)
  num_joints = 15
  N = joints.shape[2]

  neck = joints[:,0,:]
  belly = joints[:,1,:]
  face = joints[:,2,:]
  right_shoulder = joints[:,3,:]
  left_shoulder = joints[:,4,:]
  right_hip = joints[:,5,:]
  left_hip = joints[:,6,:]
  right_elbow = joints[:,7,:]
  left_elbow = joints[:,8,:]
  right_knee = joints[:,9,:]
  left_knee = joints[:,10,:]
  right_wrist = joints[:,11,:]
  left_wrist = joints[:,12,:]
  right_ankle = joints[:,13,:]
  left_ankle = joints[:,14,:]
  
  joints -= np.tile(joints[:,1,:],[num_joints,1,1]).swapaxes(0,1)
 
  #determine neck-belly angle
  #feature one will be absolute angle of reference_axis from y_axis
  reference_axis = compute_angle(joints[:,0,:],np.tile([0,1],[N,1]).T)
  #second feature will be change in angle in reference axis between frames
  prev_ref_axis = compute_prev_feature_0d(reference_axis)

  #compute_joint_angle(pivot, child_node, base_node)
  #compute_joint_angle(parent, joints, parent's parent)
  #skeleton stores joints as parent, joints, parent's parent
  skeleton = [(neck, face, belly), (neck, right_shoulder, belly), (neck, left_shoulder, belly), \
              (belly, right_hip, neck), (belly, left_hip, neck), (right_shoulder, right_elbow, neck), \
              (left_shoulder, left_elbow, neck), (right_hip, right_knee, belly), \
              (left_hip, left_knee, belly), (right_elbow, right_wrist, right_shoulder), \
              (left_elbow, left_wrist, left_shoulder), \
              (right_knee, right_ankle, right_hip), (left_knee, left_ankle, left_hip)]

  skeleton_angles = np.zeros((2*len(skeleton),len(prev_ref_axis)))
  for i, item in enumerate(skeleton):
    skeleton_angles[i*2,:] = compute_joint_angle(item[0], item[1], item[2])
    skeleton_angles[i*2+1,:] = compute_prev_feature_0d(skeleton_angles[i*2,:])

  #encode length: reference is neck belly length and is considered to have length 1.  
  #normalize length as len/len(reference)
  #normalize difference as log(new_len/old_len)
  reference_length = compute_length(neck, belly)
  prev_ref_length = compute_prev_feature_0d_log(reference_length)

  skeleton_lengths = np.zeros((2*len(skeleton),len(prev_ref_axis)))
  for i, item in enumerate(skeleton):
    skeleton_lengths[i*2,:] = compute_length(item[0], item[1])/reference_length
    skeleton_lengths[i*2+1,:] = compute_pref_feature_0d_log(skeleton_lengths[skeleton_lengths[i*2,:]])

  #concatenate reference_axis, prev_ref_axis, skeleton_angles, skeleton_lengths, reference_length, andprev_ref_length 
  feat = np.concatenate((reference_axis, prev_ref_axis),axis=0)
  feat = np.concatenate((feat, skeleton_angles),axis=0)
  feat = np.concatenate((feat, reference_length),axis=0)
  feat = np.concatenate((feat, prev_ref_length),axis=0)
  feat = np.concatenate((feat, skeleton_lengths),axis=0)

  return feat


def compute_angle(v1,v2):
  #assume angles given as [R, N] where R is the dimension and N is num samples
  #first normalize vectors
  v1 = v1/numpy.linalg.norm(v1,axis=0)
  v2 = v2/numpy.linalg.norm(v2,axis=0)
  denom = np.sqrt(np.diagonal(np.dot(v1.T,v1))) + np.sqrt(np.diagonal(np.dot(v2.T,v2)))
  return np.arccos(np.diagonal(np.dot(v1.T,v2))/denom)  #use atan2 to get full quadrant results

def compute_prev_feature_0d(feat):
  feat2 = np.roll(feat, 1)
  feat2[0] = feat[0]
  return feat - feat2

def compute_prev_feature_0d_log(feat):
  feat2 = np.roll(feat, 1)
  feat2[0] = feat[0]
  return np.log(feat2/feat)

def compute_joint_angle(pivot, child_node, root_node):
  child_node -= pivot
  root_node -= pivot
  return compute_angle(root_node, child_node) 

def compute_length(p1, p2):
  #assumes input is 2xN
  return np.sqrt((p1[0,:]-p2[0,:])**2 + (p1[1,:]-p2[1,:])**2)











