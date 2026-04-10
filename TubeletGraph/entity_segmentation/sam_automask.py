import argparse
import glob
import multiprocessing as mp
import os
import random
import sys

import json
import os.path as osp
import numpy as np
import tqdm
import torch

import pycocotools.mask as MaskUtils
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))  # add proj dir to path
from utils import load_yaml_file


def bmask_to_rle(binary_mask):
    assert binary_mask.dtype == bool, "Expecting binary mask"
    assert binary_mask.ndim == 2, "Expecting 2D mask"

    rle = MaskUtils.encode(np.asfortranarray(binary_mask))
    return {'counts': rle['counts'].decode('ascii'),
            'size': rle['size']}


def get_parser():
    parser = argparse.ArgumentParser(description="Running SAM2.1 automatic mask generator to obtain region proposals.")
    parser.add_argument("-c", "--config", default="configs/default.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("-d", '--dataset', type=str, help='Dataset to run', default='vost')
    parser.add_argument("-s", '--split', type=str, default='val', help='list of img dirs to process')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--wid', type=int, default=0, help='worker id.')
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)

    sam_cfg = cfg.entity_seg.sam_automask

    out_dir = osp.join(cfg.paths.intermdir, f'entities_{args.dataset}_sam_automask')
    os.makedirs(out_dir, exist_ok=True)

    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_model = build_sam2(sam_cfg.config, sam_cfg.model_weights, device="cuda")
    kwargs = dict(sam_cfg.kwargs) if hasattr(sam_cfg, 'kwargs') else {}
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model, **kwargs)

    with open(osp.join(data_cfg.split_dir, args.split + '.txt'), 'r') as f:
        vid_dirs = [line.strip() for line in f.readlines()]

        if args.num_workers > 1:
            random.seed(0); random.shuffle(vid_dirs)
            print('Shuffled:', ', '.join(vid_dirs[:args.num_workers]), '...')
            vid_dirs = vid_dirs[args.wid::args.num_workers]

    for vid_dir in vid_dirs:
        out_path = osp.join(out_dir, vid_dir + '.json')
        if os.path.exists(out_path):
            print(f"Skip {vid_dir} as {out_path} exists")
            continue
        frame_paths = glob.glob(os.path.join(data_cfg.image_dir, vid_dir, data_cfg.image_format))
        frame_paths = sorted(frame_paths)
        out = {}
        for path in tqdm.tqdm(frame_paths, desc=vid_dir):
            img = np.array(Image.open(path).convert("RGB"))
            masks = mask_generator.generate(img)

            # Sort by predicted_iou descending (highest quality first)
            masks = sorted(masks, key=lambda x: x['predicted_iou'], reverse=True)

            # Rasterize into non-overlapping layers (same strategy as cropformer.py):
            # higher-ranked masks overwrite lower-ranked ones
            m_H, m_W = img.shape[:2]
            mask_id = np.zeros((m_H, m_W), dtype=np.uint16)
            for rank, ann in enumerate(masks, start=1):
                seg = ann['segmentation'].astype(bool)
                mask_id[seg] = rank

            unique_mask_id = np.unique(mask_id).tolist()
            if 0 in unique_mask_id:
                unique_mask_id.remove(0)

            out[len(out)] = {ii: bmask_to_rle(mask_id == mid) for ii, mid in enumerate(unique_mask_id)}

        with open(out_path, 'w') as f:
            json.dump(out, f)
