#!/bin/sh

CUDA_VISIBLE_DEVICES=2 python $1_$2.py  --original_seq $3 --encoding_frames $4 --quantization_factor $5 --Iframe_QP $6 --Iframe_format $7 --Encoder_type $8
