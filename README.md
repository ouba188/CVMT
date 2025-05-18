# CVMT：Complex-valued mix transformer for SAR ship detection
This an official Pytorch implementation of our paper ["Complex-valued mix transformer for SAR ship detection"]. The specific details of the framework are as follows.

![img](https://github.com/RSIP-NJUPT/CVMT/blob/main/network_base.png)

To enhance portability, the CVMT codebase is entirely built on the [MMDetection](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

## Installation and Get Started(●'◡'●)ﾉ
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
git clone https://github.com/hoiliu-0801/DNTR.git
cd CVMT
python setup.py develop
```
