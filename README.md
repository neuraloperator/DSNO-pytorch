# Fast sampling of diffusion models via operator learning

## Train
Unconditional generation. 
```bash
python3 train_cifar.py --config configs/cifar10-dsno-t4.yaml --num_gpus 8
```
 
Conditional generation. 
```bash
python3 train_cond.py --config configs/imagenet64-dsno-t4.yaml --num_gpus 8
```

PS: you can add `--log` to turn on wandb for logging.

## Evaluation


## Code structure

```markdown
│   Dockerfile
│   README.md
│   train_cifar.py
│   train_cond.py
│   
├───configs
│       cifar10-dsno-t4.yaml      # example for unconditional dsno on cifar10
│       imagenet64-dsno-t4.yaml   # example for class-conditional dsno on ImageNet64
│       
├───models
│       layers.py
│       layersmt.py
│       tddpmm.py       # architecture of dsno
│       up_or_down_sampling.py
│       utils.py
│
└───utils
        dataset.py
        data_helper.py
        distributed.py
        helper.py
        loss.py
```