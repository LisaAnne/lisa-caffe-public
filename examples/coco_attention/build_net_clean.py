from __future__ import print_function
from collections import OrderedDict, Counter
import sys
sys.path.insert(0, '../coco_caption/')
from init_workspace import *
sys.path.insert(0, '../../python/')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe import to_proto
from caffe.proto import caffe_pb2


# helper function for common structures
def list_to_blob(dims):
  blob_shape = caffe_pb2.BlobShape()
  for dim in dims:
    blob_shape.dim.append(dim)
  return blob_shape

class caption_attention_model(object):

  def __init__(self, T=20, vocab_size=8801): 
    self.n = caffe.NetSpec()
    self.T = T
    self.t = 0
    self.vocab_size = vocab_size
    self.batch_size = 100
    self.buffer_size = 100
    self.sentence_length = 20
    self.lp_zero_base = [dict(lr_mult=0)]
    self.lpd_weights_ft = dict(lr_mult=0.1, decay_mult=1)      
    self.lpd_bias_ft = dict(lr_mult=0.2, decay_mult=0)    
    self.lpd_weights = dict(lr_mult=1, decay_mult=1)      
    self.lpd_bias = dict(lr_mult=2, decay_mult=0)    
    self.weight_filler_gaussian = dict(type='gaussian', std=0.01)
    self.weight_filler_constant = dict(type='uniform', min=-.08, max=0.08)
    self.bias_filler = dict(type='constant', value=0)
    self.num_vis_feature = 256
    self.num_vis_loc = 36

  def conv_relu(self,  bottom, ks, nout, stride=1, pad=0, group=1, weight_filler=None, bias_filler=None,learning_param=None):
    self.n.conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, 
                                 pad=pad, group=group, weight_filler=weight_filler, 
                                 bias_filler=bias_filler, param=learning_param)
    self.n.relu = L.ReLU(self.n.conv, in_place=True)
    return self.n.conv, self.n.relu

  def fc_relu(self, bottom, nout, weight_filler=None, bias_filler=None):
    self.n.fc = L.InnerProduct(bottom, num_output=nout, weigh_filler=weight_filler, bias_filler=bias_filler)
    self.n.relu =  L.ReLU(self.n.fc, in_place=True)
    return self.n.fc, self.n.relu 
 
  def max_pool(self, bottom, ks, stride=1):
    self.max_pool_top =  L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)
    return self.max_pool_top

  def lstm_unit(self, nout=1000, weight_filler=None):
     #lstm_unit takes previous cell, previous hidden, input for a given times step, and cont for a given time step and computes all cell units, ending with the next cell and next hidden 
 
      #inputs:
  	#botom_c: previous cell unit
          #bottom_h: previous hidden unit
          #x_transform: transformed x
          #bottom_cont: cont vector
          #bottom_static: transformed static vector
      
      t = self.t
      
      #flush hidden unit if needed
      self.n.tops['h_conted_'+str(t-1)] = L.Eltwise(self.n.tops['hidden_unit_'+str(t-1)], self.n.tops['cont_'+str(t)], operation=1, coeff_blob=True, name='flush_'+str(t))

      self.n.tops['h_transform_'+str(t)] = L.InnerProduct(self.n.tops['h_conted_'+str(t-1)], num_output=nout, bias_term=False, axis=2, param=[dict(name="W_hc")], weight_filler=weight_filler, name='h_transform_'+str(t)) 
      self.n.tops['gate_'+str(t)] = L.Eltwise(self.n.tops['h_transform_'+str(t)], self.n.tops['wt_'+str(t)], operation=1, name='compute_gate_'+str(t))
      self.n.tops['cell_unit_'+str(t)], self.n.tops['hidden_unit_'+str(t)] = L.LSTMUnit(self.n.tops['cell_unit_'+str(t-1)], self.n.tops['gate_'+str(t)], self.n.tops['cont_'+str(t)], ntop=2, name='LSTMUnit_'+str(t)) 

  def attention(self, nout, weight_filler=None, bias_filler=None):
    #\alpha_i = softmax(W_a (tanh(W_{ha} X h_{t-1}) * (tanh(W_{va} X v)) ))
    t = self.t  

    self.n.tops['hidden_unit_squeeze_'+str(t)] = L.Reshape(self.n.tops['hidden_unit_'+str(t)], shape=dict(dim=[self.batch_size, 1, 1000]), name='HiddenUnitSqueeze_'+str(t)) 
    #hidden unit
    self.n.tops['h_a_transform_'+str(t)] = L.InnerProduct(self.n.tops['hidden_unit_squeeze_'+str(t)], param=[dict(name="W_ha")], axis=2, num_output=50, weight_filler=weight_filler, bias_filler=bias_filler, name='HiddenAttentionTransform_'+str(t))
    self.n.tops['h_a_unit_'+str(t)] = L.TanH(self.n.tops['h_a_transform_'+str(t)], name='HiddenAttentionUnit_'+str(t))
    self.n.tops['h_a_unit_tile_'+str(t)] = L.Tile(self.n.tops['h_a_unit_'+str(t)], tiles=256, name='TileHiddenAttentionUnit_'+str(t))

    #attention unit
    self.n.tops['attention_unit_'+str(t)] = L.Eltwise(self.n.tops['visual_unit'], self.n.tops['h_a_unit_tile_'+str(t)], operation=0, name='AttentionUnit_'+str(t))
    self.n.tops['attention_transform_'+str(t)] = L.InnerProduct(self.n.tops['attention_unit_'+str(t)], param=[dict(name="W_ae")], num_output=self.num_vis_loc, weight_filler=weight_filler, bias_filler=bias_filler, name='AttentionTransform_'+str(t))
    self.n.tops['a_vec_'+str(t)] = L.Softmax(self.n.tops['attention_transform_'+str(t)], name='AttentionVector_'+str(t))

    #compute z feature vector
    self.n.tops['reshape_a_vec_'+str(t)] = L.Reshape(self.n.tops['a_vec_'+str(t)], shape=dict(dim=[self.batch_size, 1, self.num_vis_loc]), name='AttentionVecReshape_'+str(t)) 
    self.n.tops['a_vec_tile_'+str(t)] = L.Tile(self.n.tops['reshape_a_vec_'+str(t)], tiles=256, name='TileAttentionVector_'+str(t))
    self.n.tops['z_prod_'+str(t)] = L.Eltwise(self.n.tops['a_vec_tile_'+str(t)], self.n.tops['pool5_reshape'], operation=0, name='WeightVisualFeatures_'+str(t))
    self.n.tops['z_'+str(t)] = L.InnerProduct(self.n.tops['z_prod_'+str(t)], bias_term=False, num_output=256, weight_filler=dict(type='constant', value=1), param=self.lp_zero_base, name='SumOverFeat'+str(t))       
    self.n.tops['reshape_z_'+str(t)] = L.Reshape(self.n.tops['z_'+str(t)], shape=dict(dim=[1, self.batch_size, self.num_vis_feature]), name='Reshape_z_'+str(t))

 
  def caffenet(self, learning_param=None, weight_filler=None, bias_filler=None, in_top='data', stop_layer='fc7'):

    self.n.tops['conv1'], self.n.tops['relu1'] = self.conv_relu(self.n.tops[in_top], 11, 96, stride=4, learning_param=learning_param, weight_filler=weight_filler, bias_filler=bias_filler)
    self.n.tops['conv1'].fn.params['name'] = 'conv1'
    self.n.tops['relu1'].fn.params['name'] = 'relu1'
    if stop_layer == 'relu1':
      return
    self.n.tops['pool1'] = self.max_pool(self.n.tops['relu1'], 3, stride=2)
    self.n.tops['pool1'].fn.params['name'] = 'pool1'
    if stop_layer == 'pool1':
      return
    self.n.tops['norm1'] = L.LRN(self.n.tops['pool1'], local_size=5, alpha=1e-4, beta=0.75, name='norm1')
    if stop_layer == 'norm1':
      return
    self.n.tops['conv2'], self.n.tops['relu2'] = self.conv_relu(self.n.tops['norm1'], 5, 256, pad=2, group=2, learning_param=learning_param, weight_filler=weight_filler, bias_filler=bias_filler)
    self.n.tops['conv2'].fn.params['name'] = 'conv2'
    self.n.tops['relu2'].fn.params['name'] = 'relu2'
    if stop_layer == 'relu2':
      return
    self.n.tops['pool2'] = self.max_pool(self.n.tops['relu2'], 3, stride=2)
    self.n.tops['pool2'].fn.params['name'] = 'pool2'
    if stop_layer == 'pool2':
      return
    self.n.tops['norm2'] = L.LRN(self.n.tops['pool2'], local_size=5, alpha=1e-4, beta=0.75, name='norm2')
    if stop_layer == 'norm2':
      return
    self.n.tops['conv3'], self.n.tops['relu3'] = self.conv_relu(self.n.tops['norm2'], 3, 384, pad=1, learning_param=learning_param, weight_filler=weight_filler, bias_filler=bias_filler)
    self.n.tops['conv3'].fn.params['name'] = 'conv3'
    self.n.tops['relu3'].fn.params['name'] = 'relu3'
    if stop_layer == 'relu3':
      return
    self.n.tops['conv4'], self.n.tops['relu4'] = self.conv_relu(self.n.tops['relu3'], 3, 384, pad=1, group=2, learning_param=learning_param, weight_filler=weight_filler, bias_filler=bias_filler)
    self.n.tops['conv4'].fn.params['name'] = 'conv4'
    self.n.tops['relu4'].fn.params['name'] = 'relu4'
    if stop_layer == 'relu4':
      return
    self.n.tops['conv5'], self.n.tops['relu5'] = self.conv_relu(self.n.tops['relu4'], 3, 256, pad=1, group=2, learning_param=learning_param, weight_filler=weight_filler, bias_filler=bias_filler)
    self.n.tops['conv5'].fn.params['name'] = 'conv5'
    self.n.tops['relu5'].fn.params['name'] = 'relu5'
    if stop_layer == 'relu5':
      return
    self.n.tops['pool5'] = self.max_pool(self.n.tops['relu5'], 3, stride=2)
    self.n.tops['pool5'].fn.params['name'] = 'pool5'
    if stop_layer == 'pool5':
      return
    self.n.tops['fc6'], self.n.tops['relu6'] = fc_relu(self.n.tops['pool5'], 4096)
    self.n.tops['fc6'].fn.params['name'] = 'fc6'
    self.n.tops['relu6'].fn.params['name'] = 'relu6'
    if stop_layer == 'relu6':
      return
    self.n.tops['drop6'] = L.Dropout(self.n.tops['relu6'], in_place=True)
    self.n.tops['drop6'].fn.params['name'] = 'drop6'
    if stop_layer == 'drop6':
      return
    self.n.tops['fc7'], self.n.tops['relu7'] = fc_relu(self.n.tops['drop6'], 4096)
    self.n.tops['fc7'].fn.params['name'] = 'fc7'
    self.n.tops['relu7'].fn.params['name'] = 'relu7'
    if stop_layer == 'relu7':
      return
    self.n.tops['drop7'] = L.Dropout(self.n.tops['relu7'], in_place=True)
    self.n.tops['drop7'].fn.params['name'] = 'drop7'
    if stop_layer == 'reul7':
      return
    self.n.tops['fc8'] = L.InnerProduct(self.n.tops['drop7'], num_output=1000)
    self.n.tops['fc8'].fn.params['name'] = 'fc8'
    

  def unroll_recurrent(self):
    T = self.T

    self.n.tops['hidden_unit_0'], self.n.tops['cell_unit_0'] = L.DummyData(name='HiddenUnitInitialization', shape=[dict(dim=[1,self.batch_size, 1000]), dict(dim=[1, self.batch_size,1000])], ntop=2)

    #input to LSTM is embedded word 
    self.n.tops['wt'] = L.InnerProduct(self.n.tops['input_embed'], param=[dict(name="W_xc")], num_output=4000, axis=2, weight_filler=self.weight_filler_constant, bias_filler=self.bias_filler)
    wt_slice = L.Slice(self.n.tops['wt'], ntop=T, axis=0)
    self.rename_tops(wt_slice, 'wt_', 1, T+1, name='wtSlice')

    #vt_reshape and vt_slice are for attention model
    self.n.tops['pool5_reshape'] = L.Reshape(self.n.tops['pool5'], shape=dict(dim=[self.batch_size,self.num_vis_feature, self.num_vis_loc]), name='VisualFeatureReshape')
    self.n.tops['vt'] = L.InnerProduct(self.n.tops['pool5_reshape'], param=[dict(name="W_va")], axis=2, num_output=50, name='AttentionVisualTransform') 
    self.n.tops['visual_unit'] = L.TanH(self.n.tops['vt'], name='AttentionVisualUnit')

    for t in range(1, T+1):
      self.t = t
      self.lstm_unit(nout=4000, weight_filler=self.weight_filler_constant)
      self.attention(nout=50, weight_filler=self.weight_filler_constant, bias_filler=self.bias_filler)   
    self.n.silence_cell = L.Silence(self.n.tops['cell_unit_'+str(T)], ntop=0)

    self.n.tops['z_units'] = L.Concat(*([self.n.tops['reshape_z_'+str(t)] for t in range(1,self.T+1)]), axis=0, name='ConcatVisualFeatures')
    self.n.tops['lstm_output'] = L.Concat(*([self.n.tops['hidden_unit_'+str(t)] for t in range(1,T+1)]), axis=0, name='ConcatWordFeatures')

  def unroll_recurrent_no_attention(self):
    T = self.T

    self.n.tops['hidden_unit_0'], self.n.tops['cell_unit_0'] = L.DummyData(name='HiddenUnitInitialization', shape=[dict(dim=[1,self.batch_size, 1000]), dict(dim=[1, self.batch_size,1000])], ntop=2)

    #input to LSTM is embedded word 
    self.n.tops['wt'] = L.InnerProduct(self.n.tops['input_embed'], param=[dict(name="W_xc")], num_output=4000, axis=2, weight_filler=self.weight_filler_constant, bias_filler=self.bias_filler)
    wt_slice = L.Slice(self.n.tops['wt'], ntop=T, axis=0)
    self.rename_tops(wt_slice, 'wt_', 1, T+1, name='wtSlice')

    for t in range(1, T+1):
      self.t = t
      self.lstm_unit(nout=4000, weight_filler=self.weight_filler_constant)
    self.n.silence_cell = L.Silence(self.n.tops['cell_unit_'+str(T)], ntop=0)

    self.n.tops['lstm_output'] = L.Concat(*([self.n.tops['hidden_unit_'+str(t)] for t in range(1,T+1)]), axis=0, name='ConcatWordFeatures')
 
  def multimodal(self, multimodal_size=1024, weight_filler=None, bias_filler=None, data_in = 'multimodal_concat'):
    T = self.T
    vocab_size = self.vocab_size
    self.n.tops['multimodal'] = L.InnerProduct(self.n.tops['multimodal_concat'],num_output=multimodal_size, weight_filler=weight_filler, bias_filler=bias_filler, param=[self.lpd_weights, self.lpd_bias], axis=2, name='MultimodalUnit')
    self.n.tops['predict'] = L.InnerProduct(self.n.tops['multimodal'],num_output=vocab_size, weight_filler=weight_filler, bias_filler=bias_filler, param=[self.lpd_weights, self.lpd_bias], axis=2, name='Predict')
  
  def rename_tops(self, top_list, tag, start, end, name=None):
    for ix, t in enumerate(range(start, end)): setattr(self.n, tag+str(t), top_list[ix]) 
    if name:
      for t in range(start, end): self.n.tops[tag+str(t)].fn.params['name'] = name
           

  def attention_caption_model(self, image_data, hdf_data, test_net=True, deploy=False):
    T = self.T 
    vocab_size = self.vocab_size

    #data input
    self.n.tops['data'], self.n.tops['label'] = L.ImageData(source=image_data, batch_size=100, new_height=256, new_width=256, transform_param=dict(mirror=True, crop_size=227, mean_value=[104,117,123]), ntop=2, name='image_data_')
    self.n.tops['cont_sentence'], self.n.tops['input_sentence'], self.n.tops['target_sentence'] = L.HDF5Data(source=hdf_data, batch_size=20, ntop=3, name='sentence_data')
    self.n.silence_label = L.Silence(self.n.tops['label'], ntop=0)

    self.n.tops['input_embed'] = L.Embed(self.n.tops['input_sentence'], input_dim=vocab_size, bias_term=False, num_output=1000, name='embedding')   
 
    #caffenet
    self.caffenet(learning_param=self.lp_zero_base*2, stop_layer='pool5')

    #Prep for LSTM
    self.n.tops['cont_reshape'] = L.Reshape(self.n.tops['cont_sentence'], shape=dict(dim=[1,self.T, self.batch_size]), name='cont_reshape')
    cont_slice = L.Slice(self.n.tops['cont_reshape'], axis=1, ntop=T)
    self.rename_tops(cont_slice, 'cont_', 1, T+1, name='SliceCont')
    
    #unroll LSTM and attention model
    
    self.unroll_recurrent()
    
    #prep for multimodal
    self.n.tops['multimodal_concat'] = L.Concat(self.n.tops['z_units'], self.n.tops['lstm_output'], axis=2, name='ConcatVisualWordFeatures')
  
    #multimodal unit
    self.multimodal(multimodal_size=1024, weight_filler=self.weight_filler_constant, bias_filler=self.bias_filler)
 
    #loss and accuracy
    self.n.tops['loss'] = L.SoftmaxWithLoss(self.n.tops['predict'], self.n.tops['target_sentence'], softmax_param=dict(axis=2), loss_param=dict(ignore_label=-1))
    if test_net:
      self.n.tops['accuracy'] = L.Accuracy(self.n.tops['predict'], self.n.tops['target_sentence'], axis=2, ignore_label=-1)
  
  def lrcn_caption_model(self, image_data, hdf_data, test_net=True, deploy=False):
    T = self.T 
    vocab_size = self.vocab_size

    #data input
    self.n.tops['data'], self.n.tops['label'] = L.ImageData(source=image_data, batch_size=100, new_height=256, new_width=256, transform_param=dict(mirror=True, crop_size=227, mean_value=[104,117,123]), ntop=2, name='image_data_')
    self.n.tops['cont_sentence'], self.n.tops['input_sentence'], self.n.tops['target_sentence'] = L.HDF5Data(source=hdf_data, batch_size=20, ntop=3, name='sentence_data')
    self.n.silence_label = L.Silence(self.n.tops['label'], ntop=0)

    self.n.tops['input_embed'] = L.Embed(self.n.tops['input_sentence'], input_dim=vocab_size, bias_term=False, num_output=1000, name='embedding')   
 
    #caffenet
    self.caffenet(learning_param=self.lp_zero_base*2, stop_layer='fc7')

    self.n.tops['fc8'] = L.InnerProduct(self.n.tops['fc7'], num_ouput=1000, weight_filler=self.weight_filler_gaussian, bias_filler = self.bias_filler, name='fc8') 

    #Prep for LSTM
    self.n.tops['cont_reshape'] = L.Reshape(self.n.tops['cont_sentence'], shape=dict(dim=[1,self.T, self.batch_size]), name='cont_reshape')
    cont_slice = L.Slice(self.n.tops['cont_reshape'], axis=1, ntop=T)
    self.rename_tops(cont_slice, 'cont_', 1, T+1, name='SliceCont')
    
    # unroll lstm 
    self.unroll_recurrent_no_attention()
  
    #multimodal unit
    self.multimodal(multimodal_size=1024, weight_filler=self.weight_filler_constant, bias_filler=self.bias_filler)
 
    #loss and accuracy
    self.n.tops['loss'] = L.SoftmaxWithLoss(self.n.tops['predict'], self.n.tops['target_sentence'], softmax_param=dict(axis=2), loss_param=dict(ignore_label=-1))
    if test_net:
      self.n.tops['accuracy'] = L.Accuracy(self.n.tops['predict'], self.n.tops['target_sentence'], axis=2, ignore_label=-1)
  
def make_attention_net(data_tag, prototxt_tag, test_net=True):

  save_file = 'attention_nets/attention_build_clean_%s.prototxt' %prototxt_tag
  image_data = data_home_dir + '%s_aligned_20_batches/image_list.with_dummy_labels.txt' %data_tag
  sentence_data= data_home_dir + '%s_aligned_20_batches/hdf5_chunk_list.txt' %data_tag

  n_attention = caption_attention_model(20, 8801)
  n_attention.attention_caption_model(image_data, sentence_data, test_net) 

  with open(save_file, 'w') as f:
    print(n_attention.n.to_proto(), file=f)

def make_lrcn_net(data_tag, prototxt_tag, test_net=True):

  save_file = 'attention_nets/lrcn_build_clean_%s.prototxt' %prototxt_tag
  image_data = data_home_dir + '%s_aligned_20_batches/image_list.with_dummy_labels.txt' %data_tag
  sentence_data= data_home_dir + '%s_aligned_20_batches/hdf5_chunk_list.txt' %data_tag

  n_attention = caption_attention_model(20, 8801)
  n_attention.attention_lrcn_model(image_data, sentence_data, test_net) 

  with open(save_file, 'w') as f:
    print(n_attention.n.to_proto(), file=f)

if __name__ == '__main__':
    make_net('train', 'train-on-train', False)
    make_net('train', 'test-on-train', True)
    make_net('val', 'test-on-val', True)
