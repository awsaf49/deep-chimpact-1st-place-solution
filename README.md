# [Deep Chimpact](https://www.drivendata.org/competitions/82/competition-wildlife-video-depth-estimation/page/390/)
> Depth Estimation for Wildlife Conservation (1st place solution)

![image](https://user-images.githubusercontent.com/36858976/138281204-c3cbcb77-11ca-448b-a693-cb3cfa3c5181.png)

## Hardware requirements
* GPU (model or N/A):   8x NVIDIA Tesla V100
* Memory (GB):   8 x 32GB
* OS: Amazon Linux
* CUDA Version : 11.0
* Driver Version : 450.119.04
* CPU RAM : 128 GiB
* DISK : 2 TB

## Software requirements
Required software are listed on [requirements.txt](https://github.com/awsaf49/deep-chimpact-1st-place-solution/blob/main/requirements.txt). Please install all the dependencies before executing the pipeline.

## How to run
### Data preparation
First, the training and testing data should be downloaded from the competition website. Then run [prepare_data.py](https://github.com/awsaf49/deep-chimpact-1st-place-solution/blob/main/prepare_data.py) with appropriate arguments which are explained below: 


#### prepare_data.py
- **--data-dir** directory for raw data (unprocessed videos)
- **--save-dir** directory to save processed data (images extracted from videos)
- **--debug** uses only 10 videos for processing if this mode id used
- **--infer-only** generates images only for test videos

### Training
Run [train.py](https://github.com/awsaf49/deep-chimpact-1st-place-solution/blob/main/train.py) to train each of the 5 final models using appropriate arguments.

#### train.py
- **--cfg** config file path
- **--debug** trains only with a small portion of the entire files
- **--model-name** name of the model
- **--img-size** image size. e.g. --img-size 576 1024
- **--batch-size** batch size
- **--selected-folds** selected folds for training. e.g. --selected-folds 0 1 2 
- **--all-data** use all data for training. No validation data
- **--ds-path** dataset path
- **--output-dir** path to save model weights and necessary files

### Prediction
Run [predict_soln.py](https://github.com/awsaf49/deep-chimpact-1st-place-solution/blob/main/predict.py) in order to predict on test images.


#### predict.py
- **--cfg** config file path
- **--ckpt-cfg** config file for already given checkpoints. If new models are to be evaluated, `--cfg` should be altered accordingly.
- **--debug** trains only with a small portion of the entire files
- **--output-dir** output folder to to save submission file
- **--tta** number of TTA's

## Full Pipeline
<pre>
!python3 prepare_data.py --data-dir data/raw

!python3 train.py --model-name 'ECA_NFNetL2' --img-size 360 640 --batch-size 32
!python3 train.py --model-name 'ECA_NFNetL2' --img-size 450 800 --batch-size 24
!python3 train.py --model-name 'ECA_NFNetL2' --img-size 576 1024 --batch-size 12
!python3 train.py --model-name 'ECA_NFNetL2' --img-size 720 1280 --batch-size 8
!python3 train.py --model-name 'ECA_NFNetL2' --img-size 900 1600 --batch-size 4

!python3 train.py --model-name 'ResNest200' --img-size 360 640 --batch-size 16
!python3 train.py --model-name 'ResNest200' --img-size 576 1024 --batch-size 8

!python3 train.py --model-name 'EfficientNetV2M' --img-size 450 800 --batch-size 24
!python3 train.py --model-name 'EfficientNetV2M' --img-size 576 1024 --batch-size 12

!python3 train.py --model-name 'EfficientNetB7' --img-size 360 640 --batch-size 32
!python3 train.py --model-name 'EfficientNetB7' --img-size 450 800 --batch-size 24

python predict.py
</pre>


## Infer Pipeline
```
python prepare_data.py
python predict_solu.py
```