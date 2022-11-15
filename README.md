This repository contains the code accompanying [this video](https://www.youtube.com/watch?v=8NDX74oG3ZU). The video also refers to a ROS package, which can be found [here](https://github.com/HEART-MET/activity_recognition_ros).

See the original source repository [here](https://github.com/piergiaj/pytorch-i3d)

### Fine tune the I3D model on your dataset

The original source repository contains models trained on the Charades dataset (see [original README](https://github.com/piergiaj/pytorch-i3d)). This repository contains the code to fine-tune the RGB Charades model on a new dataset, namely, the training dataset from the [ICSR 2022 HEART-MET Activity Recognition Challenge](https://codalab.lisn.upsaclay.fr/competitions/8189).


To train this model, update the path to the dataset, labels, and the logs directory in `run.sh` and run the script:
```
./run.sh
```

To get the results on the validation set, update the path to the validation dataset and trained checkpoint in `test.sh` and run the script. This will generate a file `submission.json`, which can be zipped and submitted to Codalab.

```
./test.sh
```

The code was tested with the following versions:
```
torch==1.9.0
pytorch-lightning==1.6.1
torchvision==0.10.0
av=8.0.2
```
