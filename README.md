# CCM-stainGAN

---

This code is for the virtual staining part of paper ***Resection-Inspired Histopathological Diagnosis of Cerebral Cavernous Malformations Using Quantitative Multiphoton Microscopy***.

## Preview

|                      2 Channels                      |                       2 Channels                       |                       3 Channels                       |
| :---------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------: |
| <img src="./figure/he-mpm.gif"  height=240 width=240> | <img src="./figure/he-mpm1.gif"  height=240 width=240> | <img src="./figure/ppb-mpm.gif"  height=240 width=240> |

## Network architecture

<img src="./figure/structure.png"  height=400 width=780>

## 1. Create Envirement:

- Python 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [PyTorch](https://pytorch.org/get-started/previous-versions/) >= 1.8.0
- Python packages:

  ```shell
  cd CCM-stainGAN
  
  conda activate CCM-stainGAN
  
  pip install -r requirements.txt
  ```

## 2. Data Preparation

- The MPM and H&E images are available, download from ([Google Drive](https://drive.google.com/drive/folders/1zua6CNu9HDC657dBz4sB0UU6Fg7PRyZY?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/148UNzbngfKQeHX7RsoMnXg?pwd=ccm1)).  Other data used in this repo are available from the corresponding author upon reasonable request. 

  **Note:** access code for `Baidu Disk` is `ccm1`.
  
- Split the images into patches.

- Place the training MPM patches to `/CCM-stainGAN/dataset/trian/A/` and the H&E patches to `/CCM-stainGAN/dataset/trian/B/`.

- Place the training MPM category patches to `/CCM-stainGAN/dataset/trian/cls_A`  and the H&E  category patches to `/CCM-stainGAN/dataset/trian/cls_B` .

- Place the testing MPM patches to `/CCM-stainGAN/dataset/test/A/` and the H&E patches to `/CCM-stainGAN/dataset/test/B/`.

- Then this repo is collected as the following form:

  ```shell
  |--CCM-stainGAN
    |--train.py
    |--utils.py
    |--data_utils.py
    |--test.py
    |--output
    |--dataset
      |--MPM2HE
        |--train
          |--A
            |--mpm_000.png
            |--mpm_001.png
            :
            |--mpm_099.png
          |--B
            |--he_000.png
            |--he_001.png
            :
            |--he_099.png
          |--cls_A
            |--he_001_0.png
            |--he_002_0.png
            :
            |--he_099_8.png
          |--cls_B
            |--mpm_001_0.png
            |--mpm_002_0.png
            :
            |--mpm_099_8.png
        |--test
          |--A
            |--mpm_000.png
            :
            |--mpm_099.png
          |--B
            |--he_000.png
            :
            |--he_099.png
  ```
  
  **Note:**
  
  The image's name of category  dataset must add category label in the end of name , which split of '_'. Such as: img_001_0.png, img_002_0.png, img_003_1.png, img_004_1.png.  The last number is the label of the dataset.
## 3. Training

(1) Download the data and split the images in the correct place.

(2) Run the following command to train the model.

- Single GPU 

  ```shell
  cd /CCM-stainGAN/
  
  # transform by CCM-stainGAN
  python train.py --gpu_ids 0 --end_epoch 100 --batch_size 4 --batch_size_cls 16 --num_classes 9 --decay_epoch 50 --threshold_A 35 --threshold_B 180 --env ccm
  
  # transform by CycleGAN
  python train_cycle.py --gpu_ids 0 --end_epoch 100 --batch_size 4 --decay_epoch 50 --env cycle
  
  # transform by UTOM
  python train_utom.py --gpu_ids 0 --end_epoch 100 --batch_size 4 --decay_epoch 50 --threshold_A 35 --threshold_B 180 --env utom
  ```

- Multi GPUs

  Please change the **gpu_ids** to adapt your device.

  ```shell
  cd /CCM-stainGAN/
  
  # transform by CCM-stainGAN
  python train.py --gpu_ids 0,1,2,3 --end_epoch 100 --batch_size 4 --batch_size_cls 16 --num_classes 9 --decay_epoch 50 --threshold_A 35 --threshold_B 180 --env ccm
  
  # transform by CycleGAN
  python train_cycle.py --gpu_ids 0,1,2,3 --end_epoch 100 --batch_size 4 --decay_epoch 50 --env cycle
  
  # transform by UTOM
  python train_utom.py --gpu_ids 0,1,2,3 --end_epoch 100 --batch_size 4 --decay_epoch 50 --threshold_A 35 --threshold_B 180 --env utom
  ```

- Multi GPUs With DDP

  We also provide the **DistributedDataParallel** version of multi GPUs script. Please change the **gpu_ids** to adapt your device.

  ```shell
  cd /CCM-stainGAN/
  
  # transform by CCM-stainGAN
  python -m torch.distributed.launch --nproc_per_node=6 --master_port 11111 train_ddp.py --gpu_ids 0,1,2,3,4,5 --end_epoch 100 --batch_size 4 --batch_size_cls 8 --num_classes 9 --decay_epoch 50 --threshold_A 35 --threshold_B 180 --env ccm
  ```

The models will be saved in `/CCM-stainGAN/output/{env}/`.

## Citation

If the repo helps you, please consider citing our works:

```shell
# CCM-stainGAN
@inproceedings{CCM-stainGAN,
  title={Resection-inspired histopathological diagnosis of cerebral cavernous malformations using quantitative multiphoton microscopy},
  author={Shu Wang and Yueying Li and Yixuan Xu and Shiwei Song and Ruolan Lin and Shuoyu Xu and Xingxin Huang and Limei Zheng and Chengcong Hu and Xinquan Sun and Feng Huang and Xingfu Wang and Jianxin Chen},
  booktitle={Theranostics},
  year={2022}
}
```
