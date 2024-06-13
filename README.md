# Language-driven Grasp Detection
This is the repository of the paper "Language-driven Grasp Detection."

## Table of contents
   1. [Installation](#installation)
   1. [Datasets](#datasets)
   1. [Training](#training)
   1. [Testing](#testing)

## Installation
- Create a virtual environment
```bash
$ conda create -n lgd python=3.9
$ conda activate lgd
```

- Install pytorch
```bash
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## Datasets
- For other datasets, please obtain following their instructions: [Cornell](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp), [Jacquard](https://jacquard.liris.cnrs.fr/), [OCID-grasp](https://github.com/stefan-ainetter/grasp_det_seg_cnn), and [VMRD](https://gr.xjtu.edu.cn/zh/web/zeuslan/dataset).
- All datasets should be include in the following hierarchy:
```
|- data/
    |- cornell
    |- grasp-anything
    |- jacquard
    |- OCID_grasp
    |- VMRD
```

## Training
We use GR-ConvNet as our default deep network. To train GR-ConvNet on different datasets, you can use the following command:
```bash
$ python -m torch.distributed.launch --nproc_per_node=<num_gpus> --use_env -m train_network_diffusion --dataset grasp-anywhere --dataset-path data/grasp-anything++/ --add-file-path data/grasp-anything++/seen --description training_grasp_anything++_lgd --use-depth 0 --seen 1 --network lgd --epochs 1000
```
Furthermore, if you want to train linguistic version of other networks, use the following command:
```bash
$ python train_network.py --dataset grasp-anywhere --dataset-path data/grasp-anything/ --add-file-path data/grasp-anything++/seen --description <description> --use-depth 0 --seen 1 --network <network_name>
```
We also provide training for other baselines, you can use the following command:
```bash
$ python train_network.py --dataset <dataset> --dataset-path <dataset> --description <your_description> --use-depth 0 --network <baseline_name>
```
For instance, if you want to train GG-CNN on Grasp-Anything++, use the following command:
```bash
$ python train_network.py --dataset grasp-anywhere --dataset-path data/grasp-anything/ --add-file-path data/grasp-anything++/seen --description training_grasp_anything++_lggcnn --use-depth 0 --seen 1 --network lggcnn
```

## Testing
For testing procedure, we can apply the similar commands to test different baselines:
```bash
$ python -m torch.distributed.launch --nproc_per_node=1 --use_env -m evaluate_diffusion --dataset grasp-anywhere --dataset-path data/grasp-anything++/ --add-file-path data/grasp-anything++/seen  --iou-eval --seen 1 --use-depth 0 --network <path_to_pretrained_network>
```
or
```bash
$ python evaluate.py --dataset grasp-anywhere --dataset-path data/grasp-anything --add-file-path data/grasp-anything++/seen --iou-eval --seen 0 --use-depth 0 --network <path_to_pretrained_network>
```
Important note: `<path_to_pretrained_network>` is the path to the pretrained model obtained by training procedure. Usually, the pretrained models obtained by training are stored at `logs/<timstamp>_<training_description>`. You can select the desired pretrained model to evaluate. We do not have to specify neural architecture as the codebase will automatically detect the neural architecture. Pretrained weights are available at [this link](https://drive.google.com/file/d/1_Xu4biqQMq-3eCvCurZwNdKirMSgH-Nu/view?usp=sharing).


## Acknowledgement
Our codebase is developed based on [Vuong et al.](https://github.com/andvg3/Grasp-Anything).
