# HHF
Official PyTorch implementation of paper HHF: Hashing-guided Hinge Function for Deep Hashing Retrieval

### Requirements

### Datasets

### Training
```
python retrieval --dataset [dataset] --backbone [backbone] --hash_bit [hash_bit] --alpha [alpha] --beta [beta] --delta [delta]
```
dataset: cifar10, cifar100, coco, imagenet

hash_bit: 12, 16, 24, 32, 48, 64

backbone: googlenet, resnet, alexnet

### Inference
```
python retrieval --dataset [dataset] --backbone [backbone] --hash_bit [hash_bit] --alpha [alpha] --beta [beta] --delta [delta] --test
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
