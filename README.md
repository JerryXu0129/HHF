# HHF
Official PyTorch implementation of paper HHF: Hashing-guided Hinge Function for Deep Hashing Retrieval

### Requirements
```
NVIDIA GPU + CUDA (may CuDNN) and corresponding PyTorch framework
Python 3
```
### Datasets
##### ImageNet/COCO
We recommend you to follow [https://github.com/thuml/HashNet/tree/master/pytorch#datasets](https://github.com/thuml/HashNet/tree/master/pytorch#datasets) to prepare ImageNet and COCO images.
##### CIFAR10/CIFAR100
Please run the training command with `--dataset cifar10/cifar100` directly and the cifar10/cifar100 dataset will be downloaded automatically.
### Training
```
python retrieval --dataset [dataset] --backbone [backbone] --method [method] --hash_bit [hash_bit]
```
Arguments (default value)
```
--dataset:        dataset                                         [(imagenet), cifar10, cifar100, coco]
--backbone:       backbone network for feature extracting         [(googlenet), resnet]
--method:         baseline method                                 [(anchor), NCA, DHN]
--hash_bit:       length of hash bits                             [(48), or any positive integer that ≤ 256]
```
Other optional arguments (default value)
```
--alpha:          a hyper-parameter to control the gradient of the metric loss                              [(16), or any positive float value]
--beta:           a hyper-parameter to balance the contribution of metric loss and quantization loss        [(0.001), or any positive float value]
--delta:          a relaxation hyper-parameter to alleviate the overfitting problem                         [(0.2), or any positive float value that < 1]
--batch_size      the size of a mini-batch                                                                  [(85), or any positive integer]
```
### Inference
Add `--test` after the training command. Make sure there is a corresponding `.ckpt` file in the `./result/` directory.
```
python retrieval --dataset [dataset] --backbone [backbone] --method [method] --hash_bit [hash_bit] --test
```
### Performance

### Citation
If you use this method or this code in your research, please cite as:
```
@misc{xu2022hhf,
      title={HHF: Hashing-guided Hinge Function for Deep Hashing Retrieval}, 
      author={Chengyin Xu and Zenghao Chai and Zhengzhuo Xu and Hongjia Li and Qiruyi Zuo and Lingyu Yang and Chun Yuan},
      year={2022},
      eprint={2112.02225},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
