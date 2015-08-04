import pickle as pkl
import h5py
import sys
sys.path.insert(0,'../../python/')
import caffe
caffe.set_mode_gpu()
import numpy as np

weights = 'snapshots/attributes_JJ100_NN300_VB100_iter_50000.caffemodel'
image_list = 'utils_trainAttributes/attributes_JJ100_NN300_VB100_imageList_val.txt'
h5_labels = 'utils_trainAttributes/attributes_JJ100_NN300_VB100_val.h5'
model = 'deploy_attributes.prototxt'
attribute_list = 'attribute_lists/attributes_JJ100_NN300_VB100.pkl' 
save_accuracies = 'attribute_lists/attributes_accuracies_JJ100_NN300_VB100_wF1.pkl'

#set up caffe
net = caffe.Net(model, weights, caffe.TEST)

#set up transformer
shape = (256, 3, 227, 227)
transformer = caffe.io.Transformer({'data': shape})
transformer.set_raw_scale('data', 255)
image_mean = [103.939, 116.779, 128.68]
channel_mean = np.zeros((3,227,227))
for channel_index, mean_val in enumerate(image_mean):
  channel_mean[channel_index, ...] = mean_val
transformer.set_mean('data', channel_mean)
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_transpose('data', (2, 0, 1))

#read image list
images = (open(image_list, 'rb')).readlines()

#read groundtruth labels
f = h5py.File(h5_labels, 'r')
gt_labels = np.array(f['labels'])
f.close()

#read attribute_list
attribute_list = pkl.load(open(attribute_list,'rb'))

batch_size = 25
root_image_folder = '/y/lisaanne/coco/images/val2014/'
accuracy = np.zeros((gt_labels.shape[1],))
tp = np.zeros((gt_labels.shape[1],))
fp = np.zeros((gt_labels.shape[1],))
fn = np.zeros((gt_labels.shape[1],))
for ix in range(0, len(images), batch_size):
  if ix % 100 == 0:
    sys.stdout.write('\rOn image %d/%d.' %(ix, len(images)))
    sys.stdout.flush()
  end_idx = min(ix + batch_size, len(images))
  num_images = end_idx - ix
  epsilon = (1./gt_labels.shape[0])
  image_paths = images[ix:end_idx]
  gt_labels_chunk = gt_labels[ix:end_idx,:]
  input_images = [] 
  for image in image_paths:
    input_im = caffe.io.load_image(root_image_folder + image.split(' ')[0].replace('\n',''))
    if not ((input_im.shape[0] == 256) and (input_im.shape[1] == 256)):
      input_im = caffe.io.resize_image(input_im, (256, 256))
    input_images.append(input_im)
  input_images = caffe.io.oversample(input_images, [227,227])
  caffe_in = np.zeros(np.array(input_images.shape)[[0,3,1,2]], dtype=np.float32)
  for iy, c_in in enumerate(input_images):
    caffe_in[iy] = transformer.preprocess('data', c_in)
  net.blobs['data'].reshape(caffe_in.shape[0], caffe_in.shape[1], caffe_in.shape[2], caffe_in.shape[3])
  out = net.forward_all(data=caffe_in)
  prob_final = np.zeros((num_images, gt_labels.shape[1]))
  for iz in range(0, num_images):
    prob_final[iz] = np.mean(out['prob'][iz*10:iz*10+10],0)
  prob_final = (prob_final > 0.5).astype(int)
  #should probably do fscore and not accuracy
  label_check = gt_labels_chunk - prob_final
  label_check = (label_check == 0).astype(int)
  accuracy += np.sum(epsilon*label_check, 0)

  tp_new = ((prob_final == 1) & (gt_labels_chunk == 1)).astype(int)
  fp_new = ((prob_final == 1) & (gt_labels_chunk == 0)).astype(int)
  fn_new = ((prob_final == 0) & (gt_labels_chunk == 1)).astype(int)
  tp += np.sum(epsilon*tp_new, 0)
  fp += np.sum(epsilon*fp_new, 0)
  fn += np.sum(epsilon*fp_new, 0)

sys.stdout.write('\n')

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

accuracies = {}
accuracies['accuracy'] = accuracy
accuracies['f1'] = f1

#pkl.dump(accuracies, open(save_accuracies, 'wb'))
print "Wrting accuracy to %s.\n" %(save_accuracies)

for ix, att in enumerate(attribute_list):    
  print 'Accuracy for %s is %f and f1 score is %f.\n' %(att, accuracy[ix], f1[ix])
print 'Mean accuracy is %f.\n' %np.mean(accuracy)
     
