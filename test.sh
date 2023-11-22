#!/bin/sh

python test.py \
  --batch_size=8 \
  --checkpoint=/path/to/checkpoints/epoch-34.ckpt \
  --validation_root=/path/to/test_set \
  --validation_labels=non_existent_path \ # if labels are not available, point to a non-existent path
