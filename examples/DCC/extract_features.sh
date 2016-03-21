#!/bin/bash

#imagenet
model=prototxt/train_classifiers_deploy.imagenet.prototxt
model_weights=classification_models/alex_multilabel_FT_iter_50000.caffemodel
imagenet_images=utils/imageList/test_images_rebuttal.txt

python dcc.py --model $model \
              --model_weights $model_weights \
              --imagenet_images $imagenet_images \
              --extract_features

