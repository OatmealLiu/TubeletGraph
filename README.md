# Tracking and Understanding Object Transformations
### [Project Page](https://tubelet-graph.github.io/) | [Paper](https://arxiv.org/abs/2511.04678) | [Video](https://youtu.be/FOs0BEd5-NY)

Official PyTorch implementation for the NeurIPS 2025 paper: "Tracking and Understanding Object Transformations".

<a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/></a>  

![](assets/teaser.png)

## TODOs (By 12/2)
- [x] Expand and polish [VOST-TAS](https://github.com/YihongSun/TubeletGraph/tree/main/VOST-TAS) documentations and visualizations - Done (10/31)
- [ ] Expand and polish main code documentations
- [ ] Add quick demo from input to all predictions


## ⚙️ Installation
The code is tested with `python=3.10`, `torch==2.7.0+cu126` and `torchvision==0.22.0+cu126` on a RTX A6000 GPU.
```
git clone --recurse-submodules https://github.com/YihongSun/TubeletGraph/
cd TubeletGraph/
conda create -n tubeletgraph python=3.10 -y
conda activate tubeletgraph
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
bash thirdparty/setup_ckpts.sh

# Install SAM2 with multi-mask predictions
cd thirdparty/sam2
pip install -e .
pip install -e ".[notebooks]"
python setup.py build_ext --inplace
cd ../..

# Install CropFormer
cd thirdparty
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2 --no-build-isolation
ln -s "$(pwd)"/Entity/Entityv2/CropFormer detectron2/projects/CropFormer
cd detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops
bash make.sh
cd ../../../../../../../..
# conda install -c conda-forge libstdcxx-ng # resolves libstdc++ version mismatch if needed 

# Install FC-CLIP
cd thirdparty/fc-clip
pip install -r requirements.txt
cd ../..
```


## 🔮 Quick Start

🔹 TODO: add script & notebook for single video inference.

## 📊 Evaluations
### VOST

Please first download [VOST](https://www.vostdataset.org/) and update the corresponding paths in [configs/default.yaml](configs/default.yaml).

🔹 To compute dataset-wise predictions, please run the following lines. 
```
python TubeletGraph/run.py -c configs/default.yaml -d vost -s val -m Ours [--gpus 0 1 2 3]
```

🔹 To evaluate tracking / state graph performances, please run the following lines. 
```
python3 eval/eval_tracking.py -c configs/default.yaml -p vost-val-Ours
python3 eval/eval_state_graph.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1
```
| Data-Split-Method | J | J_S | J_M | J_L | P | R | J(tr) | J(tr)_S | J(tr)_M | J(tr)_L | P(tr) | R(tr) |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|vost-val-Ours(†)| 50.9 | 41.3 | 53.0 | 68.6 | 68.1 | 63.7 | 36.7 | 23.6 | 40.2 | 60.1 | 55.2 | 47.0

| Data-Split-Method_VLM | Sem-Acc Verb | Sem-Acc Obj | Temp-Loc Pre | Temp-Loc Rec | TF Recall (SA) | TF Recall |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|vost-val-Ours_gpt-4.1(†)| 77.3 | 73.7 | 43.1 | 20.4 | 12.0 | 6.5 |

(†) We observe very minor differences compared to the results in the paper when CropFormer and FC-CLIP are integrated into the same pytorch environment as SAM2.


## 🖼️ Visualizations

🔹 To visualizing model predictions, please run the following lines. 
```
python3 eval/vis.py -c <CONFIG> -p <PRED> [-i <INSTANCE>_<OBJ_ID>]
## example
python3 eval/vis.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1  ## visualize all
python3 eval/vis.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1 -i 555_tear_aluminium_foil_1
```
- Output visualization can be found in `_vis_out/predictions/vost-val-Ours_gpt-4.1/`, where `<INSTANCE>_<OBJ_ID>.mp4` and `<INSTANCE>_<OBJ_ID>.pdf` contain the visualized object tracks and state graph, respectively.

🔹 To visualize the internally-computed spatiotemporal partition (tubelets), please run the following lines. 
```
python3 TubeletGraph/vis/tubelets.py -c <CONFIG> -d <DATASET> -m <MODEL> -i <INSTANCE>_<OBJ_ID>
## example
python3 TubeletGraph/vis/tubelets.py -c configs/default.yaml -d vost -m cropformer -i 555_tear_aluminium_foil_1
```
- Output visualization can be found at `_vis_out/tubelets/tubelets_vost_cropformer_555_tear_aluminium_foil_1.mp4` showing *input video (top-left)*, *entity segmentation (top-right)*, *initial tubelets (bottom-left)*, and *newly emergent tubelets (bottom-right)*.


## Citation
If you find our work useful in your research, please consider citing our paper:
```
@article{sun2025tracking,
  title={Tracking and Understanding Object Transformations},
  author={Sun, Yihong and Yang, Xinyu and Sun, Jennifer J and Hariharan, Bharath},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```
