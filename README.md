# dual-diffusion-dance-style-transfer

This repo holds the code to perform experiments with the dual diffusion-based dance style transfer system.

**_Abstract_**

Current training of motion style transfer systems relies on consistency losses across style domains to prereserve contents, hindering its scalable application to a large number of domains and private data. Recent image transfer works show the potential of independent training on each domain by leveraging implicit bridging between diffusion models, with the content preservation, however, limited to simple data patterns. It is structured on folders which hold the code or assets for different parts of the workflow. This paper proposes to address this by imposing biased sampling in backward diffusion while maintaining the domain independence in the training stage. We construct the bias from the source domain keyframes and apply them as the gradient of content constraints, yielding a framework with keyframe manifold constraint gradients (KMCGs).  Our validation demonstrates the success of training separate models to transfer between as many as ten dance motion styles. Comprehensive experiments find a significant improvement in preserving motion contents in comparison to baseline and ablative diffusion-based style transfer models. In addition, we perform a human study for a subjective assessment of the quality of generated dance motions. The result substantiates the advantage of KMCGs. 


----------
## Overview of the repo

It is structured into folders which hold the code or assets for different parts of the workflow, with Pytorch Lightning.

* `dataset` holds the pytorch datasets and also convenience utilities to create the datasets.
* `models` defines the models as modules. 
* `libs` holds the libs for skeleton data. 
* `data` holds the actual data.
* `scripts` holds the scripts for training and testing.
* `render` holds the mesh for visualization.

----------

## Prerequisites

The conda environment `ddst` defined in 'requirements.txt' contains the required dependencies.

```
conda create -n ddst python=3.7.16
conda install pytorch==1.12.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/rodrigo-castellon/jukemirlib.git
```

We run each experiments on locomotion with 1 x NVIDIA A100, for the dance experiemtns, we use 4 x NVIDIA A100. 

## Data
We performed our experiments on [100STYLE](https://www.ianxmason.com/100style/) locomotion data and [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html) dance data. The preprocessed dance dataset can be downloaded from [Here](https://github.com/Stanford-TML/EDGE). 

## Training
The config file is 'scripts\args.py'.  Start training by running the following command:

```
accelerate launch --main_process_port=25148 scripts/train.py <hparams>
```

Example: For training for locking dance:
```
accelerate launch --main_process_port=25148 scripts/train.py 
--batch_size 128 \
--epochs 10000 \
--force_reload \
--data_sub gLO \
--exp_name exp_gLO \
--save_inverval 1000
```

## Transfer

```
python scripts/transfer.py --force_reload --source <source_domain> --target <target_domain> --checkpoint_source <source_model> --checkpoint_target <target_model>
```

Example: For transfer from locking dance to waacking dance:
```
python scripts/transfer.py \
--force_reload  \
--source gLO \
--target gWA \
--checkpoint_source runs/train/exp_gLO/weights/train-10000.pt \
--checkpoint_target runs/train/exp_gWA/weights/train-10000.pt \


## Acknowlegement
This implementation is based on / inspired by https://github.com/Stanford-TML/EDGE, https://github.com/suxuann/ddib, and https://github.com/HJ-harry/MCG_diffusion. 

## Citation
