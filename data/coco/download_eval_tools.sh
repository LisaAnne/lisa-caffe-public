#!/usr/bin/env bash

OUTFILE=coco_caption_eval.zip
wget --no-check-certificate https://github.com/jeffdonahue/coco-caption/archive/master.zip -O $OUTFILE
unzip $OUTFILE
mv coco-caption-master coco-caption-eval
