import argparse
import evaluate_lstm
import glob
import h5py
import os

parser = argparse.ArgumentParser()
#Model to evaluate
parser.add_argument("--model", type=str, default=None)
#split for ucf101 dataset
parser.add_argument("--split", type=int, default=1)
#folder in which to save extracted features; features will be deleted at the end unless --save_features flag included at run time
parser.add_argument("--save_folder", type=str, default='extracted_features')
#path for image frames (flow or RGB)
parser.add_argument("--im_path", type=str, default='frames')
#deploy prototxt for extracting features
parser.add_argument("--deploy_im", type=str, default='deploy_lstm_im.prototxt')
#deploy prototxt for evaluating lstm
parser.add_argument("--deploy_lstm", type=str, default='deploy_lstm_lstm.prototxt')

#add flow tag if evaluating flow images
parser.add_argument("--flow", dest='flow', action='store_true')
parser.set_defaults(flow=False)
#add save_features if you would like to save features after extraction
parser.add_argument("--save_features", dest='save_features', action='store_true')
parser.set_defaults(save_features=False)

args = parser.parse_args()

if not args.model:
  raise Exception("Must input trained model for evaluation") 
if not os.path.isdir(args.save_folder):
  print 'Creating save folder %s.' %args.save_folder
  os.mkdir(args.save_folder)

#extract features
evaluate_lstm.extract_features(args.model, flow=args.flow, split=args.split, save_folder=args.save_folder, im_path=args.im_path)

#create txt file which contains h5 files
h5Files = glob.glob('%s/*.h5' %args.save_folder)
h5_txt_name = 'h5_files.%s.txt' %args.save_folder.split('/')[0]
h5_txt = open(h5_txt_name, 'w')
for h in h5Files:
  h5_txt.writelines('%s\n' %h)
h5_txt.close()

#determine accuracy
evaluate_lstm.evaluate_lstm(args.model, h5_txt_name, model=args.deploy_lstm)

#clean up
os.remove(h5_txt_name)
if not args.save_features:
  for f in os.listdir(args.save_folder):
    os.remove(os.path.join(args.save_folder, f))
  os.rmdir(args.save_folder)



