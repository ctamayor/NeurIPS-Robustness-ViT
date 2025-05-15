## Overview

Codebase for the NeurIPS submission "Your Attention Matters: to Improve Model Robustness to Noise and Spurious Correlations." This pytorch implementation is built off of the repo https://github.com/omihub777/ViT-CIFAR where the main differences are the new attention implementations, support for Imagenette and ImageNet, and image corruptions. 

## 1. Quick Start

1. **Install packages**
```sh
$git clone https://github.com/omihub777/ViT-CIFAR.git
$cd ViT-CIFAR/
$bash setup.sh
```

2. **Corrupt Images**

For C10, C100, and Imagenette, you must first generate the images using transforms.py. For ImageNet, the corruptions are generated at runtime due to the dataset size so you do not need to run transforms.py.
```sh
$python transforms.py --dataset imagenette --severity-min 5 --severity-max 6 --corruption fog
```

3. **Train ViT**
   
Results are logged in the logs folder. To change the Doubly Stochastic specific hyperparameters (epsilon and max iterations), you must go to layers.py and change it there.
```sh
$python main.py --dataset c10 --label-smoothing --autoaugment --train-corruption fog --test-corruption fog
```
