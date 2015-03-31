#need to make .txt file to create JHDMB leveldb and corresponding hdf5 labels.
import glob
import random
import scipy.io
import pickle as pkl
import skimage.io
import numpy as np
import h5py

split = 1
random.seed(10)

#get frames and shuffle
def get_videos():
  splits = glob.glob('/mnt/y/lisaanne/JHMDB/splits/*_split1.txt')
  videos_test = []
  videos_train = []
  action_hash = pkl.load(open('/mnt/y/lisaanne/JHMDB/action_hash.p','rb'))
  for s in splits:
    f = open(s, 'rb')
    lines = f.readlines()
    f.close()
    action = s.split('/')[-1].split('_test_split')[0]
    label = action_hash[action]
    for line in lines:
      video = line.split(' ')[0].split('.avi')[0]
      t = int(line.split(' ')[1])
      if t == 1:
        videos_train.append((action, video, label))
      else:
        videos_test.append((action, video, label)) 
  return videos_test, videos_train

#gets frames from each video with corresponding joints
def get_frames_with_joints(video_list):
  video_base = '/mnt/y/lisaanne/JHMDB/'
  joint_video_list = []
  for video in video_list:
    joint_frame_list = []
    frames = sorted(glob.glob('%sRename_Images/%s/%s/*.png' %(video_base, video[0], video[1])))
    joints = scipy.io.loadmat('%sjoint_positions/%s/%s/joint_positions.mat' %(video_base, video[0], video[1]))
    j = joints['pos_img']
    for i in range(0, j.shape[2]):
      frame = frames[i]
      joint_frame_list.append((video[0], video[1], frame, video[2], j[:,:,i]))
    joint_video_list.append(joint_frame_list)

  return joint_video_list

#just readjust joints with respect to entire video frame; no bb
def preprocess_joints_no_bb(joint_videos):
  joint_videos_processed = []
  for video in joint_videos:
    joint_frame_processed = []
    size_frame = []
    for frame in video:
      if not size_frame:
        img = skimage.img_as_float(skimage.io.imread(frame[2])).astype(np.float32)
        size_frame = img.shape
      #center so all pixel values range from [-1, 1] with 0 at center of frame
      joints = frame[4]
      vis = np.ones(joints.shape)
      idx0 = np.where((joints[0,:] > size_frame[1]) | (joints[0,:] < 0))
      idx1 = np.where((joints[1,:] > size_frame[0]) | (joints[1,:] < 0))
      for idx in np.array(idx0).tolist()[0]:
        vis[:,idx] = 0
      for idx in np.array(idx1).tolist()[0]:
        vis[:,idx] = 0

      joints[0,:] = (joints[0,:]-size_frame[1])/size_frame[1]
      joints[1,:] = (joints[1,:]-size_frame[0])/size_frame[0]
      vis_cat = np.concatenate((vis[0,:], vis[1,:]), axis = 1)
      joints_cat = np.concatenate((joints[0,:], joints[1,:]), axis = 1)
      joint_frame_processed.append((frame[0], frame[1], frame[2], frame[3], joints_cat, vis_cat))
    joint_videos_processed.append(joint_frame_processed)
  return joint_videos_processed

def create_frame_data(video_frame_list, train_or_test, snap):
  #first unpack all frames
  frames = []
  for video in video_frame_list:
    frames.extend(video)
  #shuffle frames
  random.shuffle(frames)
  #create txt file for leveldb
  f = open(('JHMDB_frames_%s_%s.txt' %(snap, train_or_test)), 'wb')
  hdf_lines = {}
  hdf_lines['joints'] = np.zeros((len(frames), int(frames[0][4].shape[0]), 1, 1))
  hdf_lines['vis'] = np.zeros((len(frames), int(frames[0][5].shape[0]), 1, 1))
  hdf_lines['label'] = np.zeros((len(frames),1,1,1)) 
  for i, frame in enumerate(frames):
    line = '%s %s\n' %(frame[2], frame[3])
    hdf_lines['label'][i,0,0,0] = frame[3]
    hdf_lines['joints'][i,:,0,0] = frame[4]
    hdf_lines['vis'][i,:,0,0] = frame[5]
    f.write(line)
  f.close()
  f = h5py.File('JHMDB_joints_%s_%s.h5' %(snap, train_or_test))
  for key in hdf_lines.keys():
    f.create_dataset(key, data=hdf_lines[key])
  f.close() 

# to preprocess frames
videos_test, videos_train = get_videos()
#frames saved as (action, video, frame, label, joints)
joint_videos_test = get_frames_with_joints(videos_test)
joint_videos_train = get_frames_with_joints(videos_train)
joint_videos_test = preprocess_joints_no_bb(joint_videos_test)
joint_videos_train = preprocess_joints_no_bb(joint_videos_train)
create_frame_data(joint_videos_test, 'test', 'no_bb_try1')
create_frame_data(joint_videos_train, 'train', 'no_bb_try1')















