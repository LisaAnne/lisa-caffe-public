from __future__ import print_function
from utils.config import *
import sys
sys.path.insert(0, caffe_dir + 'python/')
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import pdb
import os
import stat

base_prototxt_file = 'prototxt/'

class caffe_net(object):

  def __init__(self):
    self.n = caffe.NetSpec()
    self.silence_count = 0

  def uniform_weight_filler(self, min_value, max_value):
    return dict(type='uniform', min=min_value, max=max_value)

  def constant_filler(self, value=0):
    return dict(type='constant', value=value)
  
  def gaussian_filler(self, value=0.01):
    return dict(type='gaussian', value=value)

  def write_net(self, save_file):
    write_proto = self.n.to_proto()
    with open(save_file, 'w') as f:
      print(write_proto, file=f)
    print("Wrote net to: %s." %save_file)
  
  def named_params(self, name_list, param_list):
    assert len(name_list) == len(param_list)
    param_dicts = []
    for name, pl in zip(name_list, param_list):
      param_dict = {}
      param_dict['name'] = name 
      param_dict['lr_mult'] = pl[0]
      if len(pl) > 1:
        param_dict['decay_mult'] = pl[0]
      param_dicts.append(param_dict)
    return param_dicts

  def learning_params(self, param_list):
    param_dicts = []
    for pl in param_list:
      param_dict = {}
      param_dict['lr_mult'] = pl[0]
      if len(pl) > 1:
        param_dict['decay_mult'] = pl[0]
      param_dicts.append(param_dict)
    return param_dicts

  def conv_relu(self, bottom, ks, nout, stride=1, pad=0, group=1, 
                weight_filler=None, bias_filler=None, learning_param=None):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group,
                                weight_filler=weight_filler, bias_filler=bias_filler, param=learning_param)
    return conv, L.ReLU(conv, in_place=True)

  def fc_relu(self, bottom, nout, 
              weight_filler=None, bias_filler=None, learning_param=None):
    fc = L.InnerProduct(bottom, num_output=nout,
                        weight_filler=weight_filler, bias_filler=bias_filler, param=learning_param)
    return fc, L.ReLU(fc, in_place=True)

  def embed(self, bottom, nout, input_dim=8801, weight_filler=None, bias_filler=None, bias_term=True, axis=1, learning_param=None, propagate_down=None):
    return L.Embed(bottom, input_dim=input_dim, num_output=nout, weight_filler=weight_filler, bias_filler=bias_filler, bias_term=bias_term, param=learning_param, propagate_down=propagate_down)

  def max_pool(self, bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

  def accuracy(self, bottom_data, bottom_label, axis=1, ignore_label=-1):
    return L.Accuracy(bottom_data, bottom_label, axis=axis, ignore_label=ignore_label)

  def softmax_loss(self, bottom_data, bottom_label, axis=1, ignore_label=-1, loss_weight=1):
    return L.SoftmaxWithLoss(bottom_data, bottom_label, loss_weight=[loss_weight], loss_param=dict(ignore_label=ignore_label), softmax_param=dict(axis=axis))
  
  def softmax_per_inst_loss(self, bottom_data, bottom_label, axis=1, ignore_label=-1, loss_weight=0):
    return L.SoftmaxPerInstLoss(bottom_data, bottom_label, loss_weight=[loss_weight], loss_param=dict(ignore_label=ignore_label), softmax_param=dict(axis=axis))

  def softmax(self, bottom_data, axis=1):
    return L.Softmax(bottom_data, axis=axis)

  def python_input_layer(self, module, layer, param_str):
    tops = L.Python(module=module, layer=layer, param_str=str(param_str), ntop=len(param_str['top_names']))
    self.rename_tops(tops, param_str['top_names']) 
 
  def python_layer(self, inputs, module, layer, param_str, ntop=1):
    return L.Python(*inputs, module=module, layer=layer, param_str=str(param_str), ntop=1)

  def rename_tops(self, top_list, new_names):
    for ix, nn in enumerate(new_names): setattr(self.n, nn, top_list[ix])

  def dummy_data_layer(self,shape, filler=1):
    #shape should be a list of dimensions
    return L.DummyData(shape=[dict(dim=shape)], data_filler=[self.constant_filler(filler)], ntop=1)

  def silence(self, bottom):
    if isinstance(bottom, list):
      self.n.tops['silence_cell_'+str(self.silence_count)] = L.Silence(*bottom, ntop=0)
    else:
      self.n.tops['silence_cell_'+str(self.silence_count)] = L.Silence(bottom, ntop=0)
    self.silence_count += 1

  def make_caffenet(self, bottom, return_layer, weight_filler={}, bias_filler={}, learning_param={}):
      default_weight_filler = self.gaussian_filler() 
      default_bias_filler = self.gaussian_filler(1)
      default_learning_param = self.learning_params([[1,1],[2,0]])
      for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
        if layer not in weight_filler.keys(): weight_filler[layer] = default_weight_filler
        if layer not in bias_filler.keys(): bias_filler[layer] = default_bias_filler
        if layer not in learning_param.keys(): learning_param[layer] = default_learning_param

      self.n.tops['conv1'], self.n.tops['relu1'] = self.conv_relu(bottom, 11, 96, stride=4,
                                                             weight_filler=weight_filler['conv1'],
                                                             bias_filler=bias_filler['conv1'],
                                                             learning_param=learning_param['conv1'])
      if return_layer in self.n.tops.keys(): return
      self.n.tops['pool1'] = self.max_pool(self.n.tops['relu1'], 3, stride=2)
      if return_layer in self.n.tops.keys(): return
      self.n.tops['norm1'] = L.LRN(self.n.tops['pool1'], local_size=5, alpha=1e-4, beta=0.75)
      if return_layer in self.n.tops.keys(): return

      self.n.tops['conv2'], self.n.tops['relu2'] = self.conv_relu(self.n.tops['norm1'], 5, 256, pad=2, group=2,
                                   weight_filler=weight_filler['conv2'],
                                   bias_filler=bias_filler['conv2'],
                                   learning_param=learning_param['conv2'])
      if return_layer in self.n.tops.keys(): return
      self.n.tops['pool2'] = self.max_pool(self.n.tops['relu2'], 3, stride=2)
      if return_layer in self.n.tops.keys(): return
      self.n.tops['norm2'] = L.LRN(self.n.tops['pool2'], local_size=5, alpha=1e-4, beta=0.75)
      if return_layer in self.n.tops.keys(): return 

      self.n.tops['conv3'], self.n.tops['relu3'] = self.conv_relu(self.n.tops['norm2'], 3, 384, pad=1,
                                                             weight_filler=weight_filler['conv3'],
                                                             bias_filler=bias_filler['conv3'],
                                                             learning_param=learning_param['conv3'])
      if return_layer in self.n.tops.keys(): return 

      self.n.tops['conv4'], self.n.tops['relu4'] = self.conv_relu(self.n.tops['relu3'], 3, 384, pad=1, group=2,
                                                             weight_filler=weight_filler['conv4'],
                                                             bias_filler=bias_filler['conv4'],
                                                             learning_param=learning_param['conv4'])
      if return_layer in self.n.tops.keys(): return 

      self.n.tops['conv5'], self.n.tops['relu5'] = self.conv_relu(self.n.tops['relu4'], 3, 256, pad=1, group=2,
                                                             weight_filler=weight_filler['conv5'],
                                                             bias_filler=bias_filler['conv5'],
                                                             learning_param=learning_param['conv5'])
      if return_layer in self.n.tops.keys(): return 
      self.n.tops['pool5'] = self.max_pool(self.n.tops['relu5'], 3, stride=2)
      if return_layer in self.n.tops.keys(): return 

      self.n.tops['fc6'], self.n.tops['relu6'] = self.fc_relu(self.n.tops['pool5'], 4096,
                                                             weight_filler=weight_filler['fc6'],
                                                             bias_filler=bias_filler['fc6'],
                                                             learning_param=learning_param['fc6'])
      if return_layer in self.n.tops.keys(): return 
      self.n.tops['drop6'] = L.Dropout(self.n.tops['relu6'], in_place=True)
      if return_layer in self.n.tops.keys(): return 
      self.n.tops['fc7'], self.n.tops['relu7'] = self.fc_relu(self.n.tops['drop6'], 4096,
                                                             weight_filler=weight_filler['fc7'],
                                                             bias_filler=bias_filler['fc7'],
                                                             learning_param=learning_param['fc7'])
      if return_layer in self.n.tops.keys(): return 'relu7'
      self.n.tops['drop7'] = L.Dropout(self.n.tops['relu7'], in_place=True)
      if return_layer in self.n.tops.keys(): return  
      self.n.tops['fc8'] = L.InnerProduct(self.n.tops['drop7'], num_output=1000,  
                                          weight_filler=weight_filler['fc8'],
                                          bias_filler=bias_filler['fc8'],
                                          param=learning_param['fc8'])

  
  def lstm(self, data, markers, top_name='lstm', lstm_static=None, weight_filler=None, bias_filler=None, learning_param_lstm=None, lstm_hidden=1000):
    #default params
    if not weight_filler: weight_filler = self.uniform_weight_filler(-.08, .08)
    if not bias_filler: bias_filler = self.constant_filler(0)
    if not learning_param_lstm: learning_param_lstm = self.learning_params([[1,1],[1,1],[1,1]])

    if lstm_static:
      self.n.tops[top_name] = L.LSTM(data, markers, lstm_static, param=learning_param_lstm,
                 recurrent_param=dict(num_output=lstm_hidden, weight_filler=weight_filler, bias_filler=bias_filler))
    else: 
      self.n.tops[top_name] = L.LSTM(data, markers, param=learning_param_lstm,
                 recurrent_param=dict(num_output=lstm_hidden, weight_filler=weight_filler, bias_filler=bias_filler))
 
  def lstm_unit(self, prefix, x, cont, static=None, h=None, c=None,
         batch_size=100, timestep=0, lstm_hidden=1000,
         weight_filler=None, bias_filler=None,
         weight_lr_mult=1, bias_lr_mult=2,
         weight_decay_mult=1, bias_decay_mult=0, concat_hidden=True):

    #assume static is already transformed
    if not weight_filler:
      weight_filler = self.uniform_weight_filler(-0.08, 0.08)
    if not bias_filler:
      bias_filler = self.constant_filler(0)
    if not h:
      h = self.dummy_data_layer([1, batch_size, lstm_hidden], 1)
    if not c:
      c = self.dummy_data_layer([1, batch_size, lstm_hidden], 1)
    gate_dim=self.gate_dim

    def get_name(name):
        return '%s_%s' % (prefix, name)
    def get_param(weight_name, bias_name=None):
        w = dict(lr_mult=weight_lr_mult, decay_mult=weight_decay_mult,
                 name=get_name(weight_name))
        if bias_name is not None:
            b = dict(lr_mult=bias_lr_mult, decay_mult=bias_decay_mult,
                     name=get_name(bias_name))
            return [w, b]
        return [w]
    # gate_dim is the dimension of the cell state inputs:
    # 4 gates (i, f, o, g), each with dimension dim
    # Add layer to transform all timesteps of x to the hidden state dimension.
    #     x_transform = W_xc * x + b_c
    cont_reshape = L.Reshape(cont, shape=dict(dim=[1,1,-1]))
    x = L.InnerProduct(x, num_output=gate_dim, axis=2,
        weight_filler=weight_filler, bias_filler=bias_filler,
        param=get_param('W_xc', 'b_c'))
    setattr(self.n, get_name('%d_x_transform' %timestep), x)
    h_conted = L.Eltwise(h, cont_reshape, coeff_blob=True) 
    h = L.InnerProduct(h_conted, num_output=gate_dim, axis=2, bias_term=False,
        weight_filler=weight_filler, param=get_param('W_hc'))
    h_name = get_name('%d_h_transform' %timestep)
    if not hasattr(self.n, h_name):
        setattr(self.n, h_name, h)
    gate_input_args = x, h
    if static is not None:
        gate_input_args += (static, )
    gate_input = L.Eltwise(*gate_input_args)
    assert cont is not None
    c, h = L.LSTMUnit(c, gate_input, cont_reshape, ntop=2)
    return h, c 

  def generate_sequence(self, data, markers, top_name='predict', lstm_static=None, weight_filler=None, bias_filler=None, learning_param_lstm=None, learning_param_ip=None, lstm_hidden=1000):
    #default params
    if not weight_filler: weight_filler = self.uniform_weight_filler(-.08, .08)
    if not bias_filler: bias_filler = self.constant_filler(0)
    if not learning_param_lstm: learning_param_lstm = self.learning_params([[1,1],[1,1],[1,1]])
    if not learning_param_ip: learning_param_ip = self.learning_params([[1,1],[2,0]])

    self.n.tops['embed'] = self.embed(self.n.tops[data], lstm_hidden, input_dim=self.vocab_size, weight_filler=weight_filler, bias_term=False, learning_param=self.learning_params([[1]])) 

    self.n.tops['lstm'] = self.lstm(self.n.tops['embed'], markers, lstm_static, param=learning_param_lstm,
                 recurrent_param=dict(num_output=lstm_hidden, weight_filler=weight_filler, bias_filler=bias_filler))

    self.n.tops[top_name] = L.InnerProduct(self.n.tops['lstm'], num_output=self.vocab_size, axis=2,
                                              weight_filler=weight_filler, bias_filler=bias_filler, param=learning_param_ip) 
    return self.n.tops[top_name] 

def make_solver(save_name, train_nets, test_nets, **kwargs):

  #set default values
  parameter_dict = kwargs
  if 'test_iter' not in parameter_dict.keys(): parameter_dict['test_iter'] = 10
  if 'test_interval' not in parameter_dict.keys(): parameter_dict['test_interval'] = 1000
  if 'base_lr' not in parameter_dict.keys(): parameter_dict['base_lr'] = 0.01
  if 'lr_policy' not in parameter_dict.keys(): parameter_dict['lr_policy'] = '"step"' 
  if 'display' not in parameter_dict.keys(): parameter_dict['display'] = 10
  if 'max_iter' not in parameter_dict.keys(): parameter_dict['max_iter'] = 110000
  if 'gamma' not in parameter_dict.keys(): parameter_dict['gamma'] = 0.5
  if 'stepsize' not in parameter_dict.keys(): parameter_dict['stepsize'] = 20000
  if 'snapshot' not in parameter_dict.keys(): parameter_dict['snapshot'] = 5000
  if 'momentum' not in parameter_dict.keys(): parameter_dict['momentum'] = 0.9
  if 'weight_decay' not in parameter_dict.keys(): parameter_dict['weight_decay'] = 0.0
  if 'solver_mode' not in parameter_dict.keys(): parameter_dict['solver_mode'] = 'GPU'
  if 'random_seed' not in parameter_dict.keys(): parameter_dict['random_seed'] = 1701
  if 'average_loss' not in parameter_dict.keys(): parameter_dict['average_loss'] = 100
  if 'clip_gradients' not in parameter_dict.keys(): parameter_dict['clip_gradients'] = 10

  snapshot_prefix = 'snapshots/%s' %save_name.split('/')[-1].split('_solver.prototxt')[0]
  parameter_dict['snapshot_prefix'] = '"%s"' %snapshot_prefix
 
  write_txt = open(save_name, 'w')
  write_txt.writelines('train_net: "%s"\n' %train_nets)
  for tn in test_nets:
    write_txt.writelines('test_net: "%s"\n' %tn)
    write_txt.writelines('test_iter: %d\n' %parameter_dict['test_iter'])
  if len(test_nets) > 0:
    write_txt.writelines('test_interval: %d\n' %parameter_dict['test_interval'])

  parameter_dict.pop('test_iter')
  parameter_dict.pop('test_interval')

  for key in parameter_dict.keys():
    write_txt.writelines('%s: %s\n' %(key, parameter_dict[key]))

  write_txt.close()

  print("Wrote solver to %s." %save_name)

def make_bash_script(save_bash, solver, weights=None, gpu=2):
  write_txt = open(save_bash, 'w')
  
  write_txt.writelines('#!/usr/bin/env bash\n\n')
  write_txt.writelines('GPU_ID=%d\n' %gpu)
  if weights:
    write_txt.writelines('WEIGHTS=%s\n\n' %weights)
  write_txt.writelines("export PYTHONPATH='utils/:$PYTHONPATH'\n\n")
  if weights:
    write_txt.writelines("../../build/tools/caffe train -solver %s -weights %s -gpu %d" %(solver, weights, gpu))
  else:
    write_txt.writelines("../../build/tools/caffe train -solver %s -gpu %d" %(solver, gpu))
  write_txt.close()  

  print("Wrote bash scripts to %s." %save_bash)

  #make bash script executable
  st = os.stat(save_bash)
  os.chmod(save_bash, st.st_mode | stat.S_IEXEC)
