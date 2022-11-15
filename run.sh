python main.py \
  --train_root=/path/to/training_set \
  --train_labels=/path/to/training_labels.json \
  --model_path=models/rgb_charades.pt \
  --num_classes=20 \
  --batch_size=8 \
  --default_root_dir=logs \
  --learning_rate=0.1 \
  --max_epochs=50 \
  --gpus=1 \
#  --resume_from_checkpoint=last.ckpt
