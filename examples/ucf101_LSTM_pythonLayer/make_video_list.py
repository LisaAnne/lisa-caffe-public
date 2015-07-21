import sys
import io
import random
random.seed(5)

video_path = '/mnt/y/lisaanne/ucf101/'

if len(sys.argv) > 1:
  split = int(sys.argv[1])
else:
  split = 1

trainlist = '%sucfTrainTestlist/trainlist%02d.txt' %(video_path, split)
testlist = '%sucfTrainTestlist/testlist%02d.txt' %(video_path, split)

def create_action_hash(trainlist):
  set_txt = open(trainlist, 'r')
  set_lines = set_txt.readlines()
  action_hash = {}
  for line in set_lines:
    a = line.split(' ')[0].split('/')[0]
    l = line.split(' ')[1]
    action_hash[a] = int(l)-1
  set_txt.close()
  return action_hash

def read_set(setlist, action_hash):
  set_txt = open(setlist, 'r')
  set_lines = set_txt.readlines()
  set_videos = []
  for line in set_lines:
    a = line.split(' ')[0].split('/')[0]
    v = line.split(' ')[0].split('.avi')[0]
    l = action_hash[a]
    set_videos.append((v, l))
  set_txt.close()
  return set_videos

def write_txt_file(setlist, filename):
  f = open(filename, 'wb')
  for line in setlist:
    f.write('%s %s\n' %(line[0], line[1]))
  f.close()

action_hash = create_action_hash(trainlist)
train_set = read_set(trainlist, action_hash)
test_set = read_set(testlist, action_hash)
random.shuffle(train_set)
random.shuffle(test_set)
write_txt_file(train_set, 'ucf101_flow_split1_trainVideos.txt')
write_txt_file(test_set, 'ucf101_flow_split1_testVideos.txt')













