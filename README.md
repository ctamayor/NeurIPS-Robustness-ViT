## Overview

Codebase for the NeurIPS submission "Your Attention Matters: to Improve Model Robustness to Noise and Spurious Correlations." This work is built off of the repo https://github.com/omihub777/ViT-CIFAR, authored by Omnihub777, licensed under MIT. Our additions are the new attention implementations, support for Imagenette and ImageNet, and image corruption functionality.

## 1. Quick Start

1. **Install packages**
```sh
$bash setup.sh
```

2. **Corrupt Images**

For C10, C100, and Imagenette, you must first generate the images using transforms.py.
```sh
$python transforms.py --dataset imagenette --severity-min 5 --severity-max 6 --corruption fog
```

3. **Train ViT**
   
Results are logged in the logs folder. To change the Doubly Stochastic specific hyperparameters (epsilon and max iterations), you must go to layers.py and change it there.
```sh
$python main.py --dataset c10 --label-smoothing --autoaugment --train-corruption fog --test-corruption fog
```
