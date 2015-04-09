#!/usr/bin/env python

import os
import json

anno_dir_path = './coco/annotations'
image_root = './coco/images'
abs_image_root = os.path.abspath(image_root)
filename_pattern = 'captions_%s2014.json'
in_sets = ['train', 'val']
out_set = 'trainval'
path_pattern = '%s/%s' % (anno_dir_path, filename_pattern)

out_data = {}
for in_set in in_sets:
    filename = path_pattern % in_set
    print 'Loading input dataset from: %s' % filename
    data = json.load(open(filename, 'r'))
    for key, val in data.iteritems():
        if type(val) == list:
            if key not in out_data:
                out_data[key] = []
            out_data[key] += val
        else:
            if key not in out_data:
                out_data[key] = val
            else:
                assert out_data[key] == val
filename = path_pattern % out_set
print 'Dumping output dataset to: %s' % filename
json.dump(out_data, open(filename, 'w'))

out_ids = [str(im['id']) for im in out_data['images']]
coco_id_filename = './coco2014_cocoid.trainval.txt'
print 'Writing COCO IDs to: %s' % coco_id_filename
with open(coco_id_filename, 'w') as coco_id_file:
    coco_id_file.write('\n'.join(out_ids) + '\n')

# make a trainval dir with symlinks to all train+val images
out_dir = '%s/%s2014' % (image_root, out_set)
os.makedirs(out_dir)
print 'Writing image symlinks to: %s' % out_dir
for im in out_data['images']:
    filename = im['file_name']
    set_name = None
    for in_set in in_sets:
        if in_set in filename:
            set_name = in_set
            break
    assert set_name is not None
    in_path = '%s/%s2014/%s' % (abs_image_root, set_name, filename)
    out_path = '%s/%s' % (out_dir, filename)
    os.symlink(in_path, out_path)
