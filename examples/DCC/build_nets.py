from build_net import caffe_net
from build_net import dcc_net 
from utils.config import *
import argparse
import pdb

def build_dcc_net_param_str(batch_size, extracted_features, image_list, feature_size):
  return {'batch_size': batch_size, 'extracted_features': extracted_features, 'image_list': image_list, 'feature_size': feature_size} 

def build_label_param_str(batch_size, extracted_features, image_list, json_images, lexical_list, feature_size,  imagenet_start=0):
  return {'batch_size': batch_size, 'extracted_features': extracted_features, 'images': image_list, 'json_images': json_images, 'lexical_list': lexical_list, 'imagenet_start': imagenet_start, 'feature_size': feature_size}

def build_reward_param_str(all_objects, train_objects, vocab):
  return {'all_objects': all_objects, 'train_objects': train_objects, 'vocab': vocab}

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
    precomputed_coco_baseline = lexical_features_root + 'vgg_feats.attributes_JJ100_NN300_VB100_allObjects_coco_vgg_0111_iter_80000.train.h5' 
    precomputed_coco_rm1 = lexical_features_root + 'vgg_feats.attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel.train.h5' 
    precomputed_imagenet_rm1 = lexical_features_root + 'vgg_feats.attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.train.h5'

  if num_features == 715:
    precomputed_coco_baseline = lexical_features_root + 'vgg_feats.attributes_JJ155_NN511_VB100_coco_715_baseline_0223_iter_80000.caffemodel.train.h5' 
    precomputed_coco_rm1 = lexical_features_root + 'vgg_feats.attributes_JJ155_NN511_VB100_coco_715classes_0116_iter_80000.train.h5' 
    precomputed_imagenet_rm1 = lexical_features_root + 'vgg_feats.attributes_JJ155_NN511_VB100_clusterEight_imagenet_vgg_715classes_0114_iter_80000.caffemodel.train.h5'
  
  #build basline net 
  baseline = dcc_net.dcc(vocab, num_features)   
  baseline_base = 'dcc_coco_baseline_vgg'
  baseline_param_str = build_dcc_net_param_str(batch_size, precomputed_coco_baseline, image_list_baseline, num_features)
  baseline.build_train_caption_net(baseline_param_str, hdf5_list_baseline, '%s.%d.train.prototxt' %(baseline_base, num_features)) 
  baseline.build_deploy_caption_net('dcc_vgg.%d.deploy.prototxt' %num_features) 
  baseline.build_wtd_caption_net('dcc_vgg.%d.wtd.prototxt' %num_features) 

  solver_name = models_root + '%s.%d.solver.prototxt' %(baseline_base, num_features)
  caffe_net.make_solver(solver_name, models_root + '%s.%d.train.prototxt' %(baseline_base, num_features), [])
  caffe_net.make_bash_script('run_%s.%d.sh' %(baseline_base, num_features), solver_name, weights=pretrained_lm+'mrnn.direct_iter_110000.caffemodel')

  #build rm_coco (coco/coco)
  rm_coco = dcc_net.dcc(vocab, num_features)  
  rm_coco_base = 'dcc_coco_rm1_vgg' 
  rm_coco_param_str = build_dcc_net_param_str(batch_size, precomputed_coco_rm1, image_list_rm1, num_features)
  rm_coco.build_train_caption_net(rm_coco_param_str, hdf5_list_rm1, '%s.%d.train.prototxt' %(rm_coco_base, num_features)) 

  solver_name = models_root + '%s.%d.solver.prototxt' %(rm_coco_base, num_features)
  caffe_net.make_solver(solver_name, models_root + '%s.%d.train.prototxt' %(rm_coco_base, num_features), [])
  caffe_net.make_bash_script('run_%s.%d.sh' %(rm_coco_base, num_features), solver_name, weights=pretrained_lm+'mrnn.direct_iter_110000.caffemodel')
  
  #build rm imagenet (imnet/coco) 
  rm_imagenet = dcc_net.dcc(vocab, num_features)  
  rm_imagenet_base = 'dcc_imagenet_rm1_vgg' 
  rm_imagenet_param_str = build_dcc_net_param_str(batch_size, precomputed_imagenet_rm1, image_list_rm1, num_features)
  rm_imagenet.build_train_caption_net(rm_imagenet_param_str, hdf5_list_rm1, '%s.%d.train.prototxt' %(rm_imagenet_base, num_features)) 

  solver_name = models_root + '%s.%d.solver.prototxt' %(rm_imagenet_base, num_features)
  caffe_net.make_solver(solver_name, models_root + '%s.%d.train.prototxt' %(rm_imagenet_base, num_features), [])
  caffe_net.make_bash_script('run_%s.%d.sh' %(rm_imagenet_base, num_features), solver_name, weights=pretrained_lm+'mrnn.direct_iter_110000.caffemodel')

  #build rm oodLM (imnet/surf) 
  rm_oodLM = dcc_net.dcc(eightyk_vocab, num_features)  
  oodLM_base = 'dcc_oodLM_rm1_vgg' 
  rm_oodLM_param_str = build_dcc_net_param_str(eightyk_batch_size, precomputed_imagenet_rm1, eightyk_image_list_rm1, num_features)
  rm_oodLM.build_train_caption_net(rm_oodLM_param_str, eightyk_hdf5_list_rm1, '%s.%d.train.prototxt' %(oodLM_base, num_features)) 
  rm_oodLM.build_deploy_caption_net('dcc_vgg.80k.%d.deploy.prototxt' %num_features) 
  rm_oodLM.build_wtd_caption_net('dcc_vgg.80k.%d.wtd.prototxt' %num_features) 

  solver_name = models_root + '%s.%d.solver.prototxt' %(oodLM_base, num_features)
  caffe_net.make_solver(solver_name, models_root + '%s.%d.train.prototxt' %(oodLM_base, num_features), [])
  caffe_net.make_bash_script('run_%s.%d.im2txt.sh' %(oodLM_base, num_features), solver_name, weights=pretrained_lm+'mrnn.lm.direct_imtextyt_lr0.01_iter_120000.caffemodel')
  caffe_net.make_bash_script('run_%s.%d.surf.sh' %(oodLM_base, num_features), solver_name, weights=pretrained_lm+'mrnn.lm.direct_surf_lr0.01_iter_120000.caffemodel')

def build_dcc_net_reinforce(num_features):
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

  precomputed_coco_baseline = lexical_features_root + 'vgg_feats.attributes_JJ100_NN300_VB100_allObjects_coco_vgg_0111_iter_80000.train.h5' 
  precomputed_coco_rm1 = lexical_features_root + 'vgg_feats.attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel.train.h5' 
  precomputed_imagenet_rm1 = lexical_features_root + 'vgg_feats.attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.train.h5'

  #reinforce parameters
  json_images='utils/json_images/imageJson_train_JJ155_NN511_VB100.json'
  image_list_coco='utils/imageList/imageTrain_coco.txt'
  lexical_list_all='utils/lexicalList/lexicalList_parseCoco_JJ100_NN300_VB100.txt'
  lexical_list_train='utils/lexicalList/lexicalList_JJ100_NN300_VB100_rmEightCoco1.txt'
 
  #build coco rm1 net 
  rm_coco = dcc_net.dcc(vocab, num_features)   
  rm_coco_base = 'dcc_reinforce_coco_rm1_vgg'
  rm_coco_param_str = build_dcc_net_param_str(batch_size, precomputed_coco_rm1, image_list_rm1, num_features)
  rm_coco_label_param_str = build_label_param_str(batch_size, precomputed_coco_rm1, image_list_coco, json_images, lexical_list_all, 471)
  reward_param_str = build_reward_param_str(lexical_list_all, lexical_list_train, vocab)
  rm_coco.build_caption_net_reinforce(rm_coco_param_str, rm_coco_label_param_str, hdf5_list_rm1, reward_param_str, '%s.%d.train.prototxt' %(rm_coco_base, num_features)) 
  rm_coco.build_deploy_caption_net('dcc_reinforce_vgg.%d.deploy.prototxt' %num_features) 
  rm_coco.build_wtd_caption_net('dcc_reinforce_vgg.%d.wtd.prototxt' %num_features, unroll=True) 

  solver_name = models_root + '%s.%d.solver.prototxt' %(rm_coco_base, num_features)
  caffe_net.make_solver(solver_name, models_root + '%s.%d.train.prototxt' %(rm_coco_base, num_features), [], 
             **{'base_lr': 0.001})
  caffe_net.make_bash_script('run_%s.sh' %rm_coco_base, solver_name, weights=caption_weights_root+'dcc_unroll_coco_baseline_vgg.471.train.dcc_coco_rm1_vgg.471.solver.prototxt_iter_110000.caffemodel', gpu=0)

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

