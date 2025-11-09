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

cd thirdparty/sam2
pip install -e .
pip install -e ".[notebooks]"
cd ../..

cd thirdparty
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2 --no-build-isolation
ln -s "$(pwd)"/Entity/Entityv2/CropFormer detectron2/projects/CropFormer
cd detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops
bash make.sh
cd ../../../../../../../..
# conda install -c conda-forge libstdcxx-ng # resolves libstdc++ version mismatch if needed 

bash thirdparty/setup_ckpts.sh
```
- Please see [INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) from the original SAM 2 repository for FAQs on potential issues and solutions.
- Please install [FC-CLIP](https://github.com/bytedance/fc-clip/tree/2b0bbe213070d44da9182530fa2e826fef03f974) with a separate conda environments according to their documentations.

And update the corresponding paths in [configs/default.yaml](configs/default.yaml) for CropFormer and FC-CLIP, accordingly.


## 🔮 Predictions
Computing entities (region proposals)
```
python3 TubeletGraph/entity_segmentation/cropformer.py -c <CONFIG> -d <DATASET> -s <SPLIT> --num_workers <N> --wid <I>
## example
conda activate cropformer      ## requires separation installation
python3 TubeletGraph/entity_segmentation/cropformer.py -c configs/default.yaml -d vost -s val
```

Computing tubelets 
```
python3 TubeletGraph/tubelet/compute_tubelets_sam.py -c <CONFIG> -d <DATASET> -s <SPLIT> --num_workers <N> --wid <I>
## example
python3 TubeletGraph/tubelet/compute_tubelets_sam.py -c configs/default.yaml -d vost -s val
```

Computing semantic similarity 
```
python3 TubeletGraph/semantic_sim/compute_sim_fcclip.py -c <CONFIG> -d <DATASET> -s <SPLIT> -t <TUBELET_NAME> --num_workers <N> --wid <I>
## example
conda activate fcclip           ## requires separation installation
python3 TubeletGraph/semantic_sim/compute_sim_fcclip.py -c configs/default.yaml -d vost -s val -t tubelets_vost_cropformer
```

Compute predictions
```
python3 TubeletGraph/get_prediction.py -c <CONFIG> -d <DATASET> -s <SPLIT> -m <METHOD>
## example
python3 TubeletGraph/get_prediction.py -c configs/default.yaml -d vost -s val -m Ours
```

Obtain state graph description
```
python3 TubeletGraph/vlm/prompt_vlm.py -c <CONFIG> -p <PRED>
## example
python3 TubeletGraph/vlm/prompt_vlm.py -c configs/default.yaml -p vost-val-Ours
```

## 📊 Evaluations
Compute tracking performances
```
python3 eval/eval.py -c <CONFIG> -p <PRED>
## example
python3 eval/eval.py -c configs/default.yaml -p vost-val-Ours
```

Compute state-graph performances
```
python3 eval/compute_temploc_pr.py -c <CONFIG> -p <PRED>
python3 eval/compute_sem_acc.py -c <CONFIG> -p <PRED>
## example
python3 eval/compute_temploc_pr.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1
python3 eval/compute_sem_acc.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1
```


## 🖼️ Visualizations
Visualizing entity segmentations
```
python3 eval/vis_entities.py -c <CONFIG> -d <DATASET> -m <MODEL> -i <INSTANCE>
## example
python3 eval/vis_entities.py -c configs/default.yaml -d vost -m cropformer -i 3161_peel_banana 
```

Visualizing tubelets
```
python3 eval/vis_tubelets.py -c <CONFIG> -d <DATASET> -m <MODEL> -i <INSTANCE>_<OBJ_ID>
## example
python3 eval/vis_tubelets.py -c configs/default.yaml -d vost -m cropformer -i 3161_peel_banana_1
```

Visualizing state graphs
```
python3 eval/vis_tubelets.py -c <CONFIG> -p <PRED>
## example
python3 eval/vis_states.py -c configs/default.yaml -p vost-val-Ours_gpt-4.1
```

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
