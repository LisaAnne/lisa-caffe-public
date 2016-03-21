import sys
from train_lexical import eval_classifiers 
import argparse
import pdb 

def extract_features(args):
  eval_classifiers.extract_features(args.model, args.model_weights, args.imagenet_images, args.device, args.image_dim, args.lexical_feature, args.batch_size)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model",type=str)
  parser.add_argument("--model_weights",type=str)
  parser.add_argument("--imagenet_images",type=str)
  parser.add_argument("--lexical_feature",type=str, default='probs')

  parser.add_argument("--device",type=int, default=0)
  parser.add_argument("--image_dim",type=int, default=227)
  parser.add_argument("--batch_size",type=int, default=10)

  parser.add_argument('--extract_features', dest='extract_features', action='store_true')
  parser.set_defaults(extract_features=False)

  args = parser.parse_args()
  
  if args.extract_features:
    extract_features(args) 




