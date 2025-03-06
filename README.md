
<div style="text-align: center; ">

# <span>Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing</span>  

</div>

<div align="center">

<h3>—— CVPR 2025 ——</h3>

**Chen Liao**<sup> 1</sup>   **Yan Shen**<sup> 1, ✉</sup>   **Dan Li**<sup> 1</sup>   **Zhongli Wang**<sup> 2</sup>

<small><sup>1</sup> School of Electronic and Information Engineering, Beijing Jiaotong University, China </small> <br>
<small><sup>2</sup> School of Automation and Intelligence, Beijing Jiaotong University, China </small>

✉️ { [liaochen](mailto:liaochen@bjtu.edu.cn) | [sheny](mailto:sheny@bjtu.edu.cn) | [lidan102628](mailto:lidan102628@bjtu.edu.cn) | [zlwang](mailto:zlwang@bjtu.edu.cn) } @bjtu.edu.cn

</div>

## Abstract

Recently, Deep Unfolding Networks (DUNs) have achieved impressive reconstruction quality in the field of image Compressive Sensing (CS) by unfolding iterative optimization algorithms into neural networks. The reconstruction quality of DUNs depends on the learned prior knowledge, so introducing stronger prior knowledge can further improve reconstruction quality. On the other hand, pre-trained diffusion models contain powerful prior knowledge and have a solid theoretical foundation and strong scalability, but it requires a large number of iterative steps to achieve reconstruction. In this paper, we propose to use the powerful prior knowledge of pre-trained diffusion model in DUNs to achieve high-quality reconstruction with less steps for image CS. Specifically, we first design an iterative optimization algorithm named Diffusion Message Passing (DMP), which embeds a pre-trained diffusion model into each iteration process of DMP. Then, we deeply unfold the DMP algorithm into a neural network named DMP-DUN. The proposed DMP-DUN can use lightweight neural networks to achieve mapping from measurement data to the intermediate steps of the reverse diffusion process and directly approximate the divergence of the diffusion model, thereby further improving reconstruction efficiency. Extensive experiments show that our proposed DMP-DUN achieves state-of-the-art performance and requires at least only 2 steps to reconstruct the image.

## Results

![](./assets/table.png)

## Enviroment

We use Ubuntu 24.04 and NVIDIA RTX 3090, with Python 3.10.10.

You can install the pip package by typing follows commands:

``` python -m pip install -r requirements.txt ```

We highly recommend to run in a new virtual Python environment.

## Testing DMP-DUN

### Download Models and Testsets (Easy Way)

We have written an automatic download script to facilitate the download. You can download a specific testset or model as the following command:
```
# Download Testset
python download.py --testset --testset-name <testset_name>
```

```
# Download Model
python download.py --model --model-name <model_name> --cs-ratios <cs_ratios>
```

The following command will donwload all testsets and models:
```
# Download Testset
python download.py --testset --testset-name Set11
python download.py --testset --testset-name Urban100

# Download DMP-DUN model weight
python download.py --model --model-name DMP_DUN_10step --cs-ratios 0.5,0.25,0.1,0.04,0.01
python download.py --model --model-name DMP_DUN_plus_2step --cs-ratios 0.5,0.25,0.1,0.04,0.01
python download.py --model --model-name DMP_DUN_plus_4step --cs-ratios 0.5,0.25,0.1,0.04,0.01
```

### Download Models and Testsets (Manual Way)

You can also download our models and testsets in manual way. Our official models and testsets can be download on [Modelscope](https://www.modelscope.cn/models/FengodChen/DMP-DUN/files) or [Baidu Netdisk](https://pan.baidu.com/s/1k7UJhswfXrmjFDWT81P1cg?pwd=8hjr). The full folder tree of our used datasets is as follows:
```
.
├── datasets
│   ├── Set11
│   │   └── 1
│   │       ├── barbara.tif
│   │       ├── ...
│   │       └── peppers256.tif
│   └── Urban100
│       └── image_SRF_4
│           ├── img_001_SRF_4_HR.png
│           ├── ...
│           └── img_100_SRF_4_HR.png
└── save
    ├── DMP_DUN_10step
    │   ├── C1R0.1
    │   │   ├── log
    │   │   │   ├── eval_log_2024-09-06_13-10-54_epoch-10.csv
    │   │   │   ├── eval_log_2024-09-08_04-27-14_epoch-20.csv
    │   │   │   ├── kernel_20240906131054.pkl
    │   │   │   ├── kernel_20240908042714.pkl
    │   │   │   ├── net_2024-09-08_04-27-14_epoch-20.pt
    │   │   │   ├── train_log_2024-09-06_13-10-54_epoch-10.csv
    │   │   │   └── train_log_2024-09-08_04-27-14_epoch-20.csv
    │   │   └── plot
    │   │        ├── train.png
    │   │        └── val.png
    │   ├── C1R0.25
    │   │   └── ...
    │   ├── C1R0.01
    │   │   └── ...
    │   ├── C1R0.04
    │   │   └── ...
    │   └── C1R0.5
    │   │   └── ...
    ├── DMP_DUN_plus_2step
    │   └── ... 
    └── DMP_DUN_plus_4step
        └── ... 
```
You can also use your own testset for testing, which can be plased in ```./datasets/<your own testset name>/1/<your image files>```. For example, if you want to test General100 dataset, you can place this dataset as follows:
```
.
└── datasets
    └── General100
        └── 1
            ├── im_1.bmp
            ├── ...
            └── im_100.bmp
```

### Test

You can run the following command to test:
```
python main.py --test --model-name <model_name> --cs-ratios <cs_ratios> --testset-names <testset_name> --gpu-id <gpu_id>
```

where the `<testset_name>` should be same as the path `./datasets/<testset_name>`. For example:
```
python main.py --test --model-name DMP_DUN_10step --cs-ratios 0.5,0.25,0.1,0.04,0.01 --testset-names Set11,Urban100 --gpu-id 0
python main.py --test --model-name DMP_DUN_plus_2step --cs-ratios 0.5,0.25,0.1,0.04,0.01 --testset-names Set11,Urban100 --gpu-id 0
python main.py --test --model-name DMP_DUN_plus_4step --cs-ratios 0.5,0.25,0.1,0.04,0.01 --testset-names Set11,Urban100 --gpu-id 0
```

And the output will be saved in ```test_ans.txt``` in the root path:
```
>>>>>> DMP_DUN_10step <<<<<<
[CS ratio = 0.5]
	[Set11] PSNR(dB)/SSIM = 42.99/0.9857
	[Urban100] PSNR(dB)/SSIM = 40.44/0.9827
[CS ratio = 0.25]
	[Set11] PSNR(dB)/SSIM = 37.92/0.9668
	[Urban100] PSNR(dB)/SSIM = 35.25/0.9538
[CS ratio = 0.1]
	[Set11] PSNR(dB)/SSIM = 32.51/0.9161
	[Urban100] PSNR(dB)/SSIM = 30.04/0.8857
[CS ratio = 0.04]
	[Set11] PSNR(dB)/SSIM = 28.20/0.8340
	[Urban100] PSNR(dB)/SSIM = 25.80/0.7727
[CS ratio = 0.01]
	[Set11] PSNR(dB)/SSIM = 23.32/0.6305
	[Urban100] PSNR(dB)/SSIM = 21.48/0.5671

>>>>>> DMP_DUN_plus_2step <<<<<<
[CS ratio = 0.5]
	[Set11] PSNR(dB)/SSIM = 42.06/0.9835
	[Urban100] PSNR(dB)/SSIM = 39.46/0.9795
[CS ratio = 0.25]
	[Set11] PSNR(dB)/SSIM = 37.58/0.9648
	[Urban100] PSNR(dB)/SSIM = 35.07/0.9520
[CS ratio = 0.1]
	[Set11] PSNR(dB)/SSIM = 32.63/0.9206
	[Urban100] PSNR(dB)/SSIM = 30.41/0.8922
[CS ratio = 0.04]
	[Set11] PSNR(dB)/SSIM = 28.25/0.8360
	[Urban100] PSNR(dB)/SSIM = 26.25/0.7858
[CS ratio = 0.01]
	[Set11] PSNR(dB)/SSIM = 23.18/0.6286
	[Urban100] PSNR(dB)/SSIM = 21.66/0.5750

>>>>>> DMP_DUN_plus_4step <<<<<<
[CS ratio = 0.5]
	[Set11] PSNR(dB)/SSIM = 42.82/0.9848
	[Urban100] PSNR(dB)/SSIM = 40.80/0.9827
[CS ratio = 0.25]
	[Set11] PSNR(dB)/SSIM = 38.29/0.9681
	[Urban100] PSNR(dB)/SSIM = 36.14/0.9584
[CS ratio = 0.1]
	[Set11] PSNR(dB)/SSIM = 33.22/0.9277
	[Urban100] PSNR(dB)/SSIM = 31.39/0.9053
[CS ratio = 0.04]
	[Set11] PSNR(dB)/SSIM = 28.67/0.8448
	[Urban100] PSNR(dB)/SSIM = 26.98/0.8035
[CS ratio = 0.01]
	[Set11] PSNR(dB)/SSIM = 23.32/0.6313
	[Urban100] PSNR(dB)/SSIM = 21.80/0.5832
```

## Training DMP-DUN
The code will automatically create, load and save the training checkpoint files at the path of `./save/<model_name>/C1R<cs_ratio>/log/`. Therefore, if you would like to train a DMP-DUN from scratch, please remove this folder manually (if it exist).

### Prepare Pretrained Guided Diffusion

We use [guided-diffusion](https://github.com/openai/guided-diffusion) as our base network. Please put the [pretrained diffusion weight](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) into ```./pretrained_guided_diffusion/``` as follows:
```
.
└── pretrained_guided_diffusion/
    └── 256x256_diffusion_uncond.pt
```

### Prepare Training and Validation Datasets

The official DMP-DUN use Coco2017 as training dataset and BSDS500 as validation datasets. You can place them into the folder `./datasets` as follows:
```
.
└── datasets
    ├── BSDS500
    │   └── val
    │       └── val
    │           ├── 101085.png
    │           ├── ...
    │           └── 97033.png
    └── Coco2017
        └── train2017
            ├── 000000000009.jpg
            ├── ...
            └── 000000581873.jpg
```

If you want to use your own dataset for training, you can also put them into `./datasets` as follows:
```
.
└── datasets
    └── <Your own dataset>
        └── 1
            ├── xxx
            ├── ...
            └── xxx
```
and you should modify the configs files in `./configs/xxx.py` to declare your dataset path and loading method (Approximately between lines 103 and 141 of the config file).

### Prepare Training Config
All the official configs are stored in `./configs/xxx.py`, you can modify this file to change some configs such as batch size, learning rate, training epoch, and etc.

### Begin Training

You can use the following command for training a new DMP-DUN:
```
python main.py --train --model-name <model_name> --cs-ratios <cs_ratios> --gpu-id <gpu_id>
```
where `<model_name>` is same as the name of config file. For example, if there exist a config file `./configs/helloworld.py`, and the `<model_name>` should be `helloworld`, i.e. `python main.py --train --model-name helloworld ...`.

For training the official DMP-DUN, you can run the following commands:
```
python main.py --train --model-name DMP_DUN_10step --cs-ratios 0.5,0.25,0.1,0.04,0.01 --gpu-id 0
python main.py --train --model-name DMP_DUN_plus_2step --cs-ratios 0.5,0.25,0.1,0.04,0.01 --gpu-id 0
python main.py --train --model-name DMP_DUN_plus_4step --cs-ratios 0.5,0.25,0.1,0.04,0.01 --gpu-id 0
```

## Citation

```
@Inproceedings{liao2025:DmpDun,
  author    = {Chen Liao, Yan Shen, Dan Li, Zhongli Wang},
  title     = {Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```