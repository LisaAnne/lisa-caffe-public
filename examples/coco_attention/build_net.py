from __future__ import print_function
import sys
sys.path.insert(0, '../../python/')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe import to_proto
from caffe.proto import caffe_pb2

# helper function for common structures

#to share params: L.Layer(..., param=[dict(name="w"), dict(name="b")])

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, weight_filler=None, bias_filler=None,learning_param=None):
  conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, 
                               group=group, weight_filler=weight_filler, bias_filler=bias_filler, 
                               param=learning_param, )
  return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, weight_filler=None, bias_filler=None):
  fc = L.InnerProduct(bottom, num_output=nout, weigh_filler=weight_filler, bias_filler=bias_filler)
  return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
  return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def lstm_unit(bottom_c, bottom_h, x_transfrom, bottom_cont, nout, bottom_static=None):

    #inputs:
	#botom_c: previous cell unit
        #bottom_h: previous hidden unit
        #x_transform: transformed x
        #bottom_cont: cont vector
        #bottom_static: transformed static vector
    
    #flush hidden unit if needed
    h_conted = L.Eltwise(bottom_h, bottom_cont, operation=1, coeff_blob=True)
    hidden_transform = L.InnerProduct(h_conted, num_output=nout, bias_term=False, param=[dict(name="W_hc")]) 
    gate_input = L.Eltwise(bottom_h, x_transfrom, operation=1)
    new_cell, new_hidden = L.LSTMUnit(bottom_c, gate_input, bottom_cont, ntop=2) 
    return new_cell, new_hidden

def attention(hidden_unit, visual_features, nout, spatial_features=169):
  #\alpha_i = softmax(W_a (tanh(W_{ha} X h_{t-1}) * (tanh(W_{va} X v)) ))

  visual_transform = L.InnerProduct(visual_features, param=[dict(name="W_va")], num_output=50)
  visual_unit = L.TanH(visual_transform)
  hidden_transform = L.InnerProduct(hidden_unit, param=[dict(name="W_va")], num_output=50)
  hidden_unit = L.TanH(hidden_transform)
  attention_unit = L.Eltwise(visual_unit, hidden_unit, operation=0)
  attention_transform = L.InnerProduct(attention_unit, param=[dict(name="W_ae")], num_output=spatial_features)
  attention_vector = L.Softmax(attention_transform)

  return attention_vector

def list_to_blob(dims):
  blob_shape = caffe_pb2.BlobShape()
  for dim in dims:
    blob_shape.dim.append(dim)

def rename_tops(top_list, tag, start, end):
  for t in range(1, T+1): setattr(n, 'input_'+str(t), input_slice[t-1]) 
  

def attention_caption_model(image_data, hdf_data):
  T = 20
  vocab_size == 8801
  n = caffe.NetSpec()

  n.tops['data'], n.tops['label'] = L.ImageData(source=image_data, batch_size=100, new_height=256, new_width=256, transform_param=dict(mirror=True, crop_size=227, mean_value=[104,117,123]), ntop=2, name='image_data')
  n.tops['cont_sentence'], n.tops['input_sentence'], n.tops['target_sentence'] = L.HDF5Data(source=hdf_data, batch_size=20, ntop=3)
  lp_zero_base=[dict(lr_mult=0)]
  
  #caffenet
  n.tops['conv1'], n.tops['relu1'] = conv_relu(n.tops['data'], 11, 96, stride=4, learning_param=lp_zero_base*2)
  n.tops['pool1'] = max_pool(n.tops['relu1'], 3, stride=2)
  n.tops['norm1'] = L.LRN(n.tops['pool1'], local_size=5, alpha=1e-4, beta=0.75)
  n.tops['conv2'], n.tops['relu2'] = conv_relu(n.tops['norm1'], 5, 256, pad=2, group=2, learning_param=lp_zero_base*2)
  n.tops['pool2'] = max_pool(n.tops['relu2'], 3, stride=2)
  n.tops['norm2'] = L.LRN(n.tops['pool2'], local_size=5, alpha=1e-4, beta=0.75)
  n.tops['conv3'], n.tops['relu3'] = conv_relu(n.tops['norm2'], 3, 384, pad=1, learning_param=lp_zero_base*2)
  n.tops['conv4'], n.tops['relu4'] = conv_relu(n.tops['relu3'], 3, 384, pad=1, group=2, learning_param=lp_zero_base*2)
  n.tops['conv5'], n.tops['relu5'] = conv_relu(n.tops['relu4'], 3, 256, pad=1, group=2, learning_param=lp_zero_base*2)
  n.tops['pool5'] = max_pool(n.tops['relu5'], 3, stride=2)

  n.tops['vt'] = L.InnerProduct(n.tops['pool5'], param=[dict(name="W_xc")], num_output=1000)
  n.tops['vt_reshape'] = L.Reshape(n.tops['vt'], shape=list_to_blob([100,256,169]))
  vt_slice = L.Slice(n.tops['vt_reshape'], axis=2, ntop=169)

  #LSTM1 recurrent + attention net
  input_slice = L.Slice(n.tops['input_sentence'], axis=1, ntop=T)
  cont_slice = L.Slice(n.tops['cont_sentence'], axis=1, ntop=T)
  for t in range(1, T+1): setattr(n, 'input_'+str(t), input_slice[t-1]) 
  for t in range(1, T+1): setattr(n, 'cont_'+str(t), cont_slice[t-1])
  for s in range(169): setattr(n, 'vt_'+str(s), vt_slice[s])
 
  n.tops['hidden_unit_0'], n.tops['cell_unit_0'] = L.DummyData(shape=list_to_blob([1,1,1000]), ntop=2)

  for t in range(1, T+1):
    n.tops['cell_unit_'+str(t)], n.tops['hidden_unit_'+str(t)] \
             = lstm_unit(n.tops['cell_unit_'+str(t-1)], n.tops['hidden_unit_'+str(t-1)], n.tops['input_'+str(t)], n.tops['cont_'+str(t)], nout=1000)
    n.tops['a_vec_'+str(t)] = attention(n.tops['hidden_unit_'+str(t-1)], n.tops['vt_reshape'], nout=50)  

    #This could probably be done with tile
    n.tops['a_vec_tile_'+str(t)] = L.Tile(n.tops['a_vec_'+str(t)], tiles=256)
    n.tops['z_'+str(t)] = L.Etlwise(n.tops['a_vec_tile_'+str(t)], n.tops['vt_reshape'])

  #multimodal
  n.tops['z_units'] = L.Concat(*([n.tops['z_'+str(t)] for t in range(1,T+1)]), axis=0)
  n.tops['hidden_units'] = L.Concat(*([n.tops['hidden_unit_'+str(t)] for t in range(1,T+1)]), axis=0)
  n.tops['multimodal_concat'] = L.Concat(n.tops['z_units'], n.tops['hidden_units'], axis=2)
  n.tops['multimodal'] = L.InnerProduct(n.tops['multimodal_concat'],num_output=1024)
  n.tops['predict'] = L.InnerProduct(n.tops['multimodal'],num_output=vocab_size)
  n.tops['loss'] = L.SoftmaxWithLoss(n.tops['predict'], n.tops['target_sentence'])
  n.tops['accuracy'] = L.Accuracy(n.tops['predict'], n.tops['target_sentence'])

  #name layers
  for top in ['data', 'label']:  n.tops[top].fn.params['name'] = 'image_data'
  for top in ['cont_sentence', 'input_sentence', 'target_sentence']: 
    n.tops[top].fn.params['name'] = 'sentence_data'
  for top in range(1,6):
    n.tops['conv'+str(top)].fn.params['name'] = 'conv'+str(top)
    n.tops['relu'+str(top)].fn.params['name'] = 'relu'+str(top)
  for top in [1,2,5]:
    n.tops['pool'+str(top)].fn.params['name'] = 'conv'+str(top)
  for top in [1,2]:
    n.tops['norm'+str(top)].fn.params['name'] = 'norm'+str(top)

  return n.to_proto() 
  
def make_net():
  save_file = 'attention.prototxt'
  image_data = '/home/lisaanne/caffe-LSTM/examples/coco_caption/h5_data/buffer_100/only_noun_sentences_noZebratrain_aligned_20_batches/image_list.with_dummy_labels.txt'
  sentence_data= '/home/lisaanne/caffe-LSTM/examples/coco_caption/h5_data/buffer_100/only_noun_sentences_noZebratrain_aligned_20_batches/hdf5_chunk_list.txt'
  with open(save_file, 'w') as f:
    print(attention_caption_model(image_data, sentence_data), file=f)

if __name__ == '__main__':
    make_net()












 
