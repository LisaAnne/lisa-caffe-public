import sys
from train_lexical import extract_classifiers 
from train_captions import transfer_weights
from train_captions import init_net 
from eval.captioner import * 
from eval import eval_sentences
import argparse
import pdb 
from utils.config import *

def extract_features(args):
  extract_classifiers.extract_features(args.image_model, args.model_weights, args.imagenet_images, args.device, args.image_dim, args.lexical_feature, args.batch_size)

def transfer(args):
  
  transfer_net = transfer_weights.transfer_net(args.language_model, args.model_weights, args.orig_attributes, args.all_attributes, args.vocab)
  eval('transfer_net.' + args.transfer_type)(args.words, args.classifiers, args.closeness_metric, args.log, num_transfer=args.num_transfer, orig_net_weights=args.orig_model) 

def generate_coco(args):
  #args.model_weights, args.image_model, args.language_model, args.vocab, args.image_list, args.precomputed_features

  model_weights = caption_weights_root + args.model_weights
  image_model = models_root + args.image_model
  language_model = models_root + args.language_model
  vocab = vocab_root + args.vocab
  if args.precomputed_features:
    precomputed_feats = lexical_features_root + args.precomputed_features
  else:
    precomputed_feats = args.precomputed_features

  image_list = open_txt(image_list_root + args.image_list)

  captioner = Captioner(model_weights, image_model, language_model, vocab, precomputed_feats=precomputed_feats,
                        prev_word_restriction=True, image_feature=args.image_feature, language_feature=args.language_feature)
  gen_json = captioner.generate_sentences(coco_images_root, image_list, temp=float('inf'), dset='coco', tag='val_val_beam1_coco')
#  gen_json = 'results/generated_sentences//_directfc7_fixIM_voc72klabel_glove_sgd_iter_45000_val_val_beam1_coco.json'
  gt_json = annotations + 'captions_%s2014.json' %args.split
  new_words = ['bus', 'bottle', 'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']
  #new_words = ['suitcase', 'zebra']
  eval_sentences.add_new_word(gt_json, gen_json, new_words)

def generate_imagenet(args):
  #args.model_weights, args.image_model, args.language_model, args.vocab, args.image_list, args.precomputed_features

  model_weights = caption_weights_root + args.model_weights
  image_model = models_root + args.image_model
  language_model = models_root + args.language_model
  vocab = vocab_root + args.vocab
  precomputed_feats = lexical_features_root + args.precomputed_features

  image_list = open_txt(image_list_root + args.image_list)

  captioner = Captioner(model_weights, image_model, language_model, vocab, precomputed_feats=precomputed_feats,
                        prev_word_restriction=True, image_feature='data', language_feature='probs')
  captioner.generate_sentences(imagenet_images_root, image_list, temp=float('inf'), dset='imagenet')

def coco_webpage(args):
  #args.model_weights, args.image_model, args.language_model, args.vocab, args.image_list, args.precomputed_features

  eval_sentences.make_coco_html()

def eval_imagenet(args):
  result_transfer = eval_sentences.make_imagenet_result_dict(generated_sentences + args.caps_transfer) 
  result_baseline = eval_sentences.make_imagenet_result_dict(generated_sentences + args.caps_baseline) 
  eval_sentences.find_successful_classes(result_transfer)

  #eval_sentences.make_imagenet_html(result_transfer, result_baseline)

def init_unroll(args):
  init_net.transfer_unrolled_net(args.orig_model, args.model_weights, args.new_model)  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_model",type=str)
  parser.add_argument("--language_model",type=str)
  parser.add_argument("--model_weights",type=str)
  parser.add_argument("--image_list", type=str)
  parser.add_argument("--imagenet_images",type=str) #extract_features
  parser.add_argument("--lexical_feature",type=str, default='probs') #name of layer to extract
  parser.add_argument("--orig_attributes",type=str, default='')
  parser.add_argument("--all_attributes",type=str, default='')
  parser.add_argument("--vocab", type=str, default='')
  parser.add_argument("--words", type=str, default='')
  parser.add_argument("--precomputed_features", type=str, default=None) #list of classifiers
  parser.add_argument("--classifiers", type=str, default='') #list of classifiers
  parser.add_argument("--closeness_metric", type=str, default='closeness_embedding')
  parser.add_argument("--transfer_type", type=str, default='direct_transfer')
  parser.add_argument("--split", type=str, default='val_val')
  parser.add_argument("--caps_baseline", type=str, default='')
  parser.add_argument("--caps_transfer", type=str, default='')

  parser.add_argument("--orig_model", type=str, default='')
  parser.add_argument("--new_model", type=str, default='')
  parser.add_argument("--language_feature", type=str, default='predict')
  parser.add_argument("--image_feature", type=str, default='data')

  parser.add_argument("--device",type=int, default=1)
  parser.add_argument("--image_dim",type=int, default=227)
  parser.add_argument("--batch_size",type=int, default=10)
  parser.add_argument("--num_transfer",type=int, default=1)

  parser.add_argument('--extract_features', dest='extract_features', action='store_true')
  parser.set_defaults(extract_features=False)
  parser.add_argument('--generate_coco', dest='generate_coco', action='store_true')
  parser.set_defaults(generate_coco=False)
  parser.add_argument('--generate_imagenet', dest='generate_imagenet', action='store_true')
  parser.set_defaults(generate_imagenet=False)
  parser.add_argument('--eval_imagenet', dest='eval_imagenet', action='store_true')
  parser.set_defaults(eval_imagenet=False)
  parser.add_argument('--transfer', dest='transfer', action='store_true')
  parser.set_defaults(transfer=False)
  parser.add_argument('--init_unroll', dest='init_unroll', action='store_true')
  parser.set_defaults(init_unroll=False)
  parser.add_argument('--coco_webpage', dest='coco_webpage', action='store_true')
  parser.set_defaults(coco_webpage=False)
  parser.add_argument('--log', dest='log', action='store_true')
  parser.set_defaults(log=False)

  args = parser.parse_args()
  
  if args.extract_features:
    extract_features(args) 

  if args.transfer:
    transfer(args)

  if args.generate_coco:
    generate_coco(args)

  if args.generate_imagenet:
    generate_imagenet(args)
  
  if args.eval_imagenet:
    eval_imagenet(args)

  if args.init_unroll:
    init_unroll(args)

  if args.coco_webpage:
    coco_webpage(args)

