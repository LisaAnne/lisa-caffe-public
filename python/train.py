#!/usr/bin/env python

import argparse
import sys
sys.path.append('./')
import caffe
caffe.set_mode_gpu()

parser = argparse.ArgumentParser()
parser.add_argument("--solver", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--weights", type=str, default=None)
args = parser.parse_args()

if not args.solver:
  raise Exception("Must indicate solver")

caffe.set_device(args.gpu)
solver = caffe.get_solver(args.solver)
if args.weights:
  solver.net.copy_from(args.weights)
  print("Copying weights from %s" %args.weights)
solver.solve()
