import json
import h5py
import random
import sys

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

image_train_json = read_json('utils_trainAttributes/imageJson_train.json')

new_objects = ['bottle', 'couch', 'luggage', 'racket', 'bus', 'microwave', 'pizza', 'suitcase', 'zebra']

coco_images = []

imagenet_images = []
for ix, im_key in enumerate(image_train_json['images']['imagenet'].keys()):
  add_im = 0     
  sys.stdout.write("\rImagenet images: im %d/%d" % (ix, len(image_train_json['images']['imagenet'].keys())))
  sys.stdout.flush()
  for no in new_objects:
     if no in im_key:
       add_im += 1
  if add_im == 1:
    imagenet_images.append(im_key)
  else:
    a = 1

print 'Number of imagenet images is %d.\n' %len(imagenet_images)

for ix, im_key in enumerate(image_train_json['images']['coco'].keys()):
  add_im = 0
  sys.stdout.write("\rCoco images: im %d/%d" % (ix, len(image_train_json['images']['coco'].keys())))
  sys.stdout.flush()
  for no in new_objects:
      if no in image_train_json['images']['coco'][im_key]['positive_label']:
          add_im += 1
  if add_im == 0:
      coco_images.append(im_key)
print '\n'

print 'Number of coco images is %d.\n' %len(coco_images)


write_txt = open('utils_trainAttributes/imageTrain_eightImagenet.txt', 'w')

write_list = []
for im in coco_images:
  write_list.append(('coco', im))
for im in imagenet_images:
  write_list.append(('imagenet', im))

random.shuffle(write_list)

for w in write_list:
  write_txt.writelines('%s %s\n' %(w[0], w[1]))

write_txt.close()






 

