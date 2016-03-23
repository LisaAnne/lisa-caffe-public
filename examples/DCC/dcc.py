import sys
from train_lexical import eval_classifiers 
from train_captions import transfer_weights
import argparse
import pdb 

def extract_features(args):
  eval_classifiers.extract_features(args.model, args.model_weights, args.imagenet_images, args.device, args.image_dim, args.lexical_feature, args.batch_size)

def transfer(args):
  
  transfer_net = transfer_weights.transfer_net(args.model, args.model_weights, args.orig_attributes, args.all_attributes, args.vocab)
  eval('transfer_net.' + args.transfer_type)(args.words, args.classifiers, args.closeness_metric, args.log) 

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model",type=str)
  parser.add_argument("--model_weights",type=str)
  parser.add_argument("--imagenet_images",type=str) #extract_features
  parser.add_argument("--lexical_feature",type=str, default='probs') #name of layer to extract
  parser.add_argument("--orig_attributes",type=str, default='')
  parser.add_argument("--all_attributes",type=str, default='')
  parser.add_argument("--vocab", type=str, default='')
  parser.add_argument("--words", type=str, default='')
  parser.add_argument("--classifiers", type=str, default='') #list of classifiers
  parser.add_argument("--closeness_metric", type=str, default='closeness_embedding')
  parser.add_argument("--transfer_type", type=str, default='direct_transfer')

  parser.add_argument("--device",type=int, default=0)
  parser.add_argument("--image_dim",type=int, default=227)
  parser.add_argument("--batch_size",type=int, default=10)

  parser.add_argument('--extract_features', dest='extract_features', action='store_true')
  parser.set_defaults(extract_features=False)
  parser.add_argument('--transfer', dest='transfer', action='store_true')
  parser.set_defaults(transfer=False)
  parser.add_argument('--log', dest='log', action='store_true')
  parser.set_defaults(log=False)

  args = parser.parse_args()
  
  if args.extract_features:
    extract_features(args) 

  if args.transfer:
    transfer(args)


