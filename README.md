This repository contains the code accompanying [this video](https://www.youtube.com/watch?v=8NDX74oG3ZU). The video also refers to a ROS package, which can be found [here](https://github.com/HEART-MET/activity_recognition_ros).

See the original source repository [here](https://github.com/piergiaj/pytorch-i3d)

### Fine tune the I3D model on your dataset

The original source repository contains models trained on the Charades dataset (see [original README](https://github.com/piergiaj/pytorch-i3d)). This repository contains the code to fine-tune the RGB Charades model on a new dataset, namely, the validation dataset from the [HEART-MET Activity Recognition challenge](https://competitions.codalab.org/competitions/30423). You can find a checkpoint for the fine-tuned model in the ROS repository mentioned above.


To train this model, update the path to the dataset, labels, and the logs directory in `run.sh` and run the script:
```
./run.sh
```

The code was tested with the following versions:
```
torch==1.9.0
pytorch-lightning==1.2.10
torchvision==0.10.0
av=8.0.3
```
