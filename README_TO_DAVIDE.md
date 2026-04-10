## Main file changes

Added SAM entity seg: "TubeletGraph/entity_segmentation/sam_automask.py"
Added OAI CLIP for semantic sim score: "TubeletGraph/semantic_sim/compute_sim_clip.py"

Entry point: "TubeletGraph/run.py"


## How to use
```bash
# FC-CLIP + SAM2 as entity segmentation module
python quick_run.py --input_dir assets/example/0334_cut_fruit_1 --input_mask assets/example/0334_cut_fruit_1_0000000.png --config configs/default_sam_entity.yaml 

# FC-CLIP + CropFormer as entity segmentation module (their default)
python quick_run.py --input_dir assets/example/0334_cut_fruit_1 --input_mask assets/example/0334_cut_fruit_1_0000000.png --config configs/default.yaml 

# OAI CLIP ViT-L/14 + SAM2 as entity segmentation module
python quick_run.py --input_dir assets/example/0334_cut_fruit_1 --input_mask assets/example/0334_cut_fruit_1_0000000.png --config configs/default_sam_entity_std_clip.yaml

# OAI CLIP ViT-L/14 + CropFormer as entity segmentation module (their default)
python quick_run.py --input_dir assets/example/0334_cut_fruit_1 --input_mask assets/example/0334_cut_fruit_1_0000000.png --config configs/default_std_clip.yaml 
```

## How to modify the config

These config can be changed in the config file by modifying:

@Entity segmentation module

- Default CropFormer
```yaml
tubelet:
  entity_method: cropformer
methods:
  Ours:
    tubelet_dirname: "tubelets_{}_cropformer_fcclip"
```

- SAM2.1
```yaml
tubelet:
  entity_method: sam_automask
methods:
  Ours:
    tubelet_dirname: "tubelets_{}_sam_automask_fcclip"
```

@Semantic Similarity computation

- Default CropFormer
```yaml
sem_sim:
  name: fcclip
methods:
  Ours:
    tubelet_dirname: "tubelets_{}_cropformer_fcclip"
```

- SAM2.1
```yaml
sem_sim:
  name: clip
methods:
  Ours:
    tubelet_dirname: "tubelets_{}_cropformer_clip"
```