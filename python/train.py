#!/usr/bin/env python

import argparse
import sys
sys.path.append('./')
import caffe
caffe.set_mode_gpu()

def train_caffe_model(solver, weights=None, gpu=0, add_paths=[]):

  for add_path in add_paths:
    sys.path.append(add_path)

  caffe.set_device(gpu)
  solver = caffe.get_solver(solver)
  if weights:
    solver.net.copy_from(weights)
    print("Copying weights from %s" %weights)
  solver.solve()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--solver", type=str, default=None)
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--weights", type=str, default=None)
  args = parser.parse_args()
  
  if not args.solver:
    raise Exception("Must indicate solver")

  train_caffe_model(args.solver, args.weights, args.gpu) 
