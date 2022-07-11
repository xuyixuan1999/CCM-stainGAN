# CCM-stainGAN：Code will be uploaded as soon as the paper is accepted.

This code is for the virtual staining part of paper ***Resection-Inspired Histopathological Diagnosis of Cerebral Cavernous Malformations Using Quantitative Multiphoton Microscopy***.


## Preview
|                      2 Channels                       |                       2 Channels                       |                       3 Channels                       |
| :---------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------: |
| <img src="./figure/he-mpm.gif"  height=240 width=240> | <img src="./figure/he-mpm1.gif"  height=240 width=240> | <img src="./figure/ppb-mpm.gif"  height=240 width=240> |

## Network architecture

<img src="./figure/structure.png"  height=400 width=720>

## 1. Create Envirement:

- Python 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- PyTorch 1.8

- Python packages:

  ```shell
  cd CCMStain-GAN 
  pip install -r requirements.txt
  ```

## 2. Data Preparation

- The MPM and H&E images are available, download from ([Google Drive](https://pytorch.org/get-started/previous-versions/) / [Baidu Disk](https://www.baidu.com)). 

  Note: access code for `Baidu Disk` is `ccm1`.

- Split the images into tiles by `data_utils.py`

- Place the training MPM tiles to `/CCMStain-GAN/dataset/trian/A/` and the H&E tiles to `/CCMStain-GAN/dataset/trian/B/`

- Place the testing MPM tiles to `/CCMStain-GAN/dataset/test/A/` and the H&E tiles to `/CCMStain-GAN/dataset/test/B/`

- Then this repo is collected as the following form:

  ```shell
  |--CCMStain-GAN
  	|--train.py
  	|--utils.py
  	|--data_utils.py
  	|--test.py
  	|--output
  	|--dataset
  		|--train
  			|--A
  				|--mpm_000_000.png
  				|--mpm_000_001.png
  				:
  				|--mpm_099_099.png
  			|--B
  				|--he_000_000.png
  				|--he_000_001.png
  				:
  				|--he_099_099.png
          |--test
          	|--A
  				|--mpm_000_000.png
  				|--mpm_000_001.png
  				:
  				|--mpm_099_099.png
  			|--B
  				|--he_000_000.png
  				|--he_000_001.png
  				:
  				|--he_099_099.png
  ```

## 3. Training

(1) Download the data and split the images in the correct place.

(2) Run the following command to train the model

```shell
cd /CCMStain-GAN/

# transform by CCMStrain-GAN
python train_cls.py --env ccm
# transform by CycleGAN
python train_cycle.py --env cycle
# transform by UTOM
python train_utom.py --env utom
```

​	Or run the following command

```shell
cd /CCMStain-GAN/

# transform by CCMStrain-GAN
bash train_cls.sh
# transform by CycleGAN
bash train_cycle.sh
# transform by UTOM
bash train_utom.sh
```

The models will be saved in `/CCMStain-GAN/output/{env}/`.

## Citation 

If the repo helps you, please consider citing our works;

```shell
# CCMStrain-GAN

```

