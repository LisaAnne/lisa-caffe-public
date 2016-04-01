from build_net import caffe_net
from build_net import dcc_net 
from utils.config import *
import argparse

def build_dcc_net_param_str(batch_size, extracted_features, image_list, feature_size):
  return {'batch_size': batch_size, 'extracted_features': extracted_features, 'image_list': image_list, 'feature_size': feature_size} 

def build_dcc_net(num_features):
  vocab = vocab_root + 'vocabulary.txt' 
  batch_size = 100
  image_list_baseline = '/home/lisaanne/caffe-LSTM/examples/coco_caption/h5_data_fixN/buffer_100/train_aligned_20_batches/image_list.with_dummy_labels.txt' 
  hdf5_list_baseline = '/home/lisaanne/caffe-LSTM/examples/coco_caption/h5_data_fixN/buffer_100/train_aligned_20_batches/hdf5_chunk_list.txt' 
  image_list_rm1 = '/home/lisaanne/caffe-LSTM/examples/coco_caption/h5_data_fixN/buffer_100/no_caption_rm_eightCluster_train_aligned_20_batches/image_list.with_dummy_labels.txt' 
  hdf5_list_rm1 = '/home/lisaanne/caffe-LSTM/examples/coco_caption/h5_data_fixN/buffer_100/no_caption_rm_eightCluster_train_aligned_20_batches/hdf5_chunk_list.txt' 

  eightyk_vocab = vocab_root + 'yt_coco_surface_80k_vocab.txt'
  eightyk_batch_size = 50
  eightyk_image_list_rm1 = '/home/lisaanne/caffe-LSTM/examples/coco_caption/h5_data_fixN_80k/buffer_50/no_caption_rm_eightCluster_train_aligned_20_batches/image_list.with_dummy_labels.txt'
  eightyk_hdf5_list_rm1 = '/home/lisaanne/caffe-LSTM/examples/coco_caption/h5_data_fixN_80k/buffer_50/no_caption_rm_eightCluster_train_aligned_20_batches/hdf5_chunk_list.txt'

  if num_features == 471:
    precomputed_coco_baseline = lexical_features_root + 'vgg_feats.attributes_JJ100_NN300_VB100_allObjects_coco_vgg_0111_iter_80000.%s.h5' 
    precomputed_coco_rm1 = lexical_features_root + 'vgg_feats.attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.%s.h5' 
    precomputed_imagenet_rm1 = lexical_features_root + 'vgg_feats.attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.%s.h5'
 
  #build basline net 
  baseline = dcc_net.dcc(vocab, num_features)   
  baseline_base = 'dcc_coco_baseline_vgg'
  baseline_param_str = build_dcc_net_param_str(batch_size, precomputed_coco_baseline, image_list_baseline, num_features)
  baseline.build_train_caption_net(baseline_param_str, hdf5_list_baseline, '%s.%d.train.prototxt' %(baseline_base, num_features)) 
  baseline.build_deploy_caption_net('dcc_vgg.%d.deploy.prototxt' %num_features) 
  baseline.build_wtd_caption_net('dcc_vgg.%d.wtd.prototxt' %num_features) 

  solver_name = '%s.%d.train.prototoxt' %(baseline_base, num_features)
  caffe_net.make_solver(baseline_base, solver_name, [])
  caffe_net.make_bash_script('run_%s.sh' %baseline_base, solver_name, weights=pretrained_lm+'mrnn.direct_iter_110000.caffemodel')

  #build rm_coco (coco/coco)
  rm_coco = dcc_net.dcc(vocab, num_features)  
  rm_coco_base = 'dcc_coco_rm1_vgg' 
  rm_coco_param_str = build_dcc_net_param_str(batch_size, precomputed_coco_rm1, image_list_rm1, num_features)
  rm_coco.build_train_caption_net(rm_coco_param_str, hdf5_list_rm1, '%s.%d.train.prototxt' %(rm_coco_base, num_features)) 

  solver_name = '%s.%d.train.prototoxt' %(rm_coco_base, num_features)
  caffe_net.make_solver(rm_coco_base, solver_name, [])
  caffe_net.make_bash_script('run_%s.sh' %rm_coco_base, solver_name, weights=pretrained_lm+'mrnn.direct_iter_110000.caffemodel')
  
  #build rm imagenet (imnet/coco) 
  rm_imagenet = dcc_net.dcc(vocab, num_features)  
  rm_imagenet_base = 'dcc_imagenet_rm1_vgg' 
  rm_imagenet_param_str = build_dcc_net_param_str(batch_size, precomputed_imagenet_rm1, image_list_rm1, num_features)
  rm_imagenet.build_train_caption_net(rm_imagenet_param_str, hdf5_list_rm1, '%s.%d.train.prototxt' %(rm_imagenet_base, num_features)) 

  solver_name = '%s.%d.train.prototoxt' %(rm_imagenet_base, num_features)
  caffe_net.make_solver(rm_imagenet_base, solver_name, [])
  caffe_net.make_bash_script('run_%s.sh' %rm_imagenet_base, solver_name, weights=pretrained_lm+'mrnn.direct_iter_110000.caffemodel')

  #build rm oodLM (imnet/surf) 
  rm_oodLM = dcc_net.dcc(eightyk_vocab, num_features)  
  oodLM_base = 'dcc_oodLM_rm1_vgg' 
  rm_oodLM_param_str = build_dcc_net_param_str(eightyk_batch_size, precomputed_imagenet_rm1, eightyk_image_list_rm1, num_features)
  rm_oodLM.build_train_caption_net(rm_oodLM_param_str, eightyk_hdf5_list_rm1, 'dcc_%s.%d.train.prototxt' %(oodLM_base, num_features)) 
  rm_oodLM.build_deploy_caption_net('dcc_vgg.80k.%d.deploy.prototxt' %num_features) 
  rm_oodLM.build_wtd_caption_net('dcc_vgg.80k.%d.wtd.prototxt' %num_features) 

  solver_name = '%s.%d.train.prototoxt' %(oodLM_base, num_features)
  caffe_net.make_solver(oodLM_base, solver_name, [])
  caffe_net.make_bash_script('run_%s.im2txt.sh' %oodLM_base, solver_name, weights=pretrained_lm+'mrnn.lm.direct_imtextyt_lr0.01_iter_120000.caffemodel')
  caffe_net.make_bash_script('run_%s.surf.sh' %oodLM_base, solver_name, weights=pretrained_lm+'mrnn.lm.direct_surf_lr0.01_iter_120000.caffemodel')

def build_dcc_net_reinforce(num_features):
  pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_features",type=int) 
  parser.add_argument('--reinforce', dest='reinforce', action='store_true')
  parser.set_defaults(reinforce=False)
  args = parser.parse_args()

  if args.reinforce:
    build_dcc_net_reinforce(args.num_features)
  else:
    build_dcc_net(args.num_features)

