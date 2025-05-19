# CVMT：Complex-valued mix transformer for SAR ship detection
This an official Pytorch implementation of our paper ["Complex-valued mix transformer for SAR ship detection"]. The specific details of the framework are as follows.

![img](https://github.com/RSIP-NJUPT/CVMT/blob/main/network_base.png)

- Notablely, CVMT is built on top of [MMDetection](https://github.com/open-mmlab/mmdetection) with an added dual-stream data loading and processing mechanism. We hope that our approach and code will be helpful for your related research.

## Installation (●'◡'●)ﾉ
Please follow the following steps for installation.

Step 1: Create a conda environment

```shell
conda create --name CVMT python=3.8
conda activate CVMT
```

Step 2: Install PyTorch 1.10.1+CU113
```shell
# CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Step 3: Install OpenMMLab codebases
```shell
#Install MMEngine and MMCV using MIM
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"

#Required packages
pip install timm
```

Step 4: Install `CVMT`
```shell
https://github.com/RSIP-NJUPT/CVMT.git
cd CVMT
python setup.py develop
```
## Get Started (ง๑ •̀_•́)ง
Get Started with single GPU

Training CVMT, for example :

```shell
python tools/train.py ./mmdet/configs/CVMT/CVMT_4scale_r50_8xb2_12e_coco.py
```

Testing CVMT, for example :
```
python tools/test.py ./mmdet/configs/CVMT/CVMT_4scale_r50_8xb2_12e_coco.py path/to/your/checkpoint
```
## Data Preparation σ`∀´)σ
- The [OpenSARShip](https://ieeexplore.ieee.org/document/8067489) & [FAIR-CSAR](https://radars.ac.cn/web/data/getData?dataType=FAIR_CSAR_en&pageType=en) or other complex-valued SAR datasets should be prepared as follows:
```
Dataset_root
├── train
│   ├── IT (Amplitude data)
│   ├── POS (Phase data)   
├── val
│   ├── IT (Amplitude data)
│   ├── POS (Phase data)
├── annotations
│   ├── annotations_train.json
│   ├── annotations_val.json
```

## Pre-Trained Models ヾ(*´∀ ˋ*)ﾉ
You can leverage the pretrained weights we provide in this work as a starting point for your own fine‑tuning.The pretrained CVMT models are available on Baidu Netdisk:

**Link:** [https://pan.baidu.com/s/1bfvJrRYLDo9U-pkqZGGWUA](https://pan.baidu.com/s/1bfvJrRYLDo9U-pkqZGGWUA)

**Extraction code:** iq95



## Performance ('ᴗ' )و

Table 1. Training Set: **FAIR-CSAR** trainval set, Testing Set: **FAIR-CSAR** test set, 12 epochs.
|Method | Backbone | mAP | mAP<sub>50</sub> | mAP<sub>75</sub> |mAP<sub>S</sub> | mAP<sub>M</sub>  | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---: |
DINO (Baseline) | R-50 | 28.9 | 67.4 | 20.1 | 29.9 | **31.4** | 
CVMT  | R-50 | **32.4** | **73.6** | **23.1** | **33.6** | 26.7 |

Table 2. Training Set: **OpenSARShip** train set, Testing Set: **OpenSARShip** test set, 36 epochs.
|Method | Backbone | mAP | mAP<sub>50</sub> | mAP<sub>75</sub> |mAP<sub>S</sub> | mAP<sub>M</sub>  | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---: |
DINO (Baseline) | R-50 | 74.6 | 88.2 | 78.6 | 74.8 | 70.0 | 
CVMT  | R-50 | **77.5** | **89.0** | **80.0** | **77.0** | **91.0** |





