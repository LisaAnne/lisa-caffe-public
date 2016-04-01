import sys
sys.path.insert(0, '../../python/')
import caffe
sys.path.insert(0, 'utils/')
import pdb
import numpy as np

def transfer_unrolled_net(orig_model_name, orig_weights, new_model_name):
  
  orig_model = caffe.Net('prototxts/' + orig_model_name, '/yy2/lisaanne/mrnn_direct/snapshots_final/' + orig_weights + '.caffemodel', caffe.TRAIN)
  new_model = caffe.Net('prototxts/' + new_model_name, caffe.TRAIN)
  
  new_model.params['predict-im'][0].data[...] = orig_model.params['predict-im'][0].data
  new_model.params['embedding1_0'][0].data[...] = orig_model.params['embedding'][0].data
  new_model.params['embedding2_0'][0].data[...] = orig_model.params['embedding2'][0].data
  new_model.params['embedding2_0'][1].data[...] = orig_model.params['embedding2'][1].data
  new_model.params['lstm1_0_x_transform'][0].data[...] = orig_model.params['lstm1'][0].data
  new_model.params['lstm1_0_x_transform'][1].data[...] = orig_model.params['lstm1'][1].data
  new_model.params['lstm1_0_h_transform'][0].data[...] = orig_model.params['lstm1'][2].data
  new_model.params['predict-lm_0'][0].data[...] = orig_model.params['predict-lm'][0].data
  new_model.params['predict-lm_0'][1].data[...] = orig_model.params['predict-lm'][1].data

  for layer in new_model.params.keys():
    for i in range(len(new_model.params[layer])):
      print "%s[%d] min weight: %f, max_weight: %f\n" %(layer, i, 
                                                 np.min(new_model.params[layer][i].data), 
                                                 np.max(new_model.params[layer][i].data))
  
  save_name = '%s.%s.caffemodel' %(new_model_name.split('.prototxt')[0], orig_weights)
  new_model.save(save_name)
  print "Saved unrolled model to %s\n" %save_name
