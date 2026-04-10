import argparse, random, glob, os, sys
import multiprocessing as mp
import os.path as osp

sys.path.insert(1, osp.join(sys.path[0], '..'))

import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import json
import pycocotools.mask as MaskUtils
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))
from utils import load_yaml_file, load_anno

import clip


def extract_mask_crop(img_rgb, mask, preprocess, padding_ratio=0.1):
    """Crop image to mask bbox, zero out background with mean pixel, preprocess for CLIP."""
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    pad_y = int(bh * padding_ratio)
    pad_x = int(bw * padding_ratio)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)

    crop = img_rgb[y0:y1+1, x0:x1+1].copy()
    crop_mask = mask[y0:y1+1, x0:x1+1]

    mean_pixel = crop[crop_mask].mean(axis=0) if crop_mask.any() else np.array([128, 128, 128], dtype=np.uint8)
    crop[~crop_mask] = mean_pixel.astype(np.uint8)

    pil_crop = Image.fromarray(crop)
    return preprocess(pil_crop).unsqueeze(0)


def get_parser():
    parser = argparse.ArgumentParser(description="Running standard CLIP to obtain clip features.")
    parser.add_argument("-c", "--config", default="configs/default.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("-d", '--dataset', type=str, help='Dataset to run', default='vost')
    parser.add_argument("-s", '--split', type=str, default='val', help='list of img dirs to process')
    parser.add_argument("-t", '--tubelet_name', type=str, default='tubelets_vost_cropformer', help='tubelet directory name')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--wid', type=int, default=0, help='worker id.')
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    my_cfg = load_yaml_file(args.config)
    data_cfg = getattr(my_cfg.datasets, args.dataset)
    clip_cfg = my_cfg.sem_sim.clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_cfg.model_name, device)

    args.tubelet_dir = osp.join(my_cfg.paths.intermdir, args.tubelet_name)
    out_dir = args.tubelet_dir.rstrip('/') + '_clip'
    os.makedirs(out_dir, exist_ok=True)

    with open(osp.join(data_cfg.split_dir, args.split+'.txt'), 'r') as f:
        split_names = [x.strip() for x in f.readlines()]

        instance_names = []
        for instance in split_names:
            init_prompt_path = sorted(glob.glob(osp.join(data_cfg.anno_dir, instance, data_cfg.anno_format)))[0]
            prompt_objs = load_anno(init_prompt_path)
            instance_names += [(instance + '_' + k + '.json', instance) for k in prompt_objs.keys()]
        assert np.all([osp.exists(osp.join(args.tubelet_dir, x)) for x,y in instance_names])

        if args.num_workers > 1:
            random.seed(0); random.shuffle(instance_names)
            print('Shuffled:', ', '.join([x for x,y in instance_names[:args.num_workers]]), '...')
            instance_names = instance_names[args.wid::args.num_workers]

    for anno_fname, video_name in instance_names:
        out_path = osp.join(out_dir, anno_fname)
        if osp.exists(out_path):
            print(f"Skip {anno_fname} as {out_path} exists")
            continue

        with open(osp.join(args.tubelet_dir, anno_fname), 'r') as f:
            load_data = json.load(f)
            all_tracks = load_data['all_tracks']
            tracked_objs = load_data['tracked_objs']
            init_tracked_objs = [obj_idx for obj_idx, obj_info in tracked_objs.items() if obj_info['init_frame_idx'] == 0]
            later_tracked_objs = [obj_idx for obj_idx, obj_info in tracked_objs.items() if obj_info['init_frame_idx'] > 0]
            prompt_obj = str(max([int(obj_idx) for obj_idx in init_tracked_objs]))
            candidate_objs = [obj_idx for obj_idx, obj_info in tracked_objs.items() if 'mm_iou' in obj_info and obj_info['mm_iou']>0]
            obj_ind_to_comp = set([prompt_obj] + candidate_objs)

        for obj_idx in later_tracked_objs:
            for metric_name in ['clip_sim', 'clip_sim_min', 'clip_sim_max', 'clip_sim_a', 'clip_sim_a_min', 'clip_sim_a_max']:
                tracked_objs[obj_idx][metric_name] = 0

        if len(candidate_objs) > 0:
            frame_paths = glob.glob(osp.join(data_cfg.image_dir, video_name, data_cfg.image_format))
            frame_paths = sorted(frame_paths)

            feat_dim = model.visual.output_dim
            out = dict()
            for ii, path in tqdm.tqdm(enumerate(frame_paths), desc=anno_fname.replace('.json', ''), total=len(frame_paths)):
                mdata = all_tracks[str(ii)]
                obj_ind_str = [k for k in mdata.keys() if k in obj_ind_to_comp]

                img_rgb = np.array(Image.open(path).convert("RGB"))

                frame_feats = {}
                for k in obj_ind_str:
                    mask = MaskUtils.decode([mdata[k]])[..., -1].astype(bool)
                    crop_tensor = extract_mask_crop(img_rgb, mask, preprocess)
                    if crop_tensor is None:
                        continue
                    with torch.no_grad():
                        feat = model.encode_image(crop_tensor.to(device)).float().cpu()
                    frame_feats[k] = feat

                out[str(ii)] = frame_feats

            frame_ind = list([int(x) for x in all_tracks.keys()]); frame_ind.remove(0); frame_ind.sort(); frame_ind = [str(x) for x in frame_ind]
            placeholder = torch.ones(1, feat_dim) * torch.inf
            query_clip_feat = torch.cat([out[frame_idx][prompt_obj] if prompt_obj in out[frame_idx] else placeholder for frame_idx in ['0'] + frame_ind])
            query_valid = query_clip_feat[:,0] < torch.inf

            query_clip_feat = F.normalize(query_clip_feat, dim=-1).T
            for obj_idx in candidate_objs:
                later_clip_feat_list = [out[frame_idx][obj_idx] for frame_idx in frame_ind if obj_idx in out[frame_idx]]
                if len(later_clip_feat_list) > 0:
                    later_clip_feats = F.normalize(torch.cat(later_clip_feat_list), dim=-1)
                    cos_sim_all = later_clip_feats @ query_clip_feat[:, query_valid]

                    init_idx = tracked_objs[obj_idx]['init_frame_idx']
                    num_valid_prior = torch.sum(query_valid[:init_idx]).item()
                    cos_sim_prior = cos_sim_all[:, :num_valid_prior] if num_valid_prior > 0 else None
                else:
                    cos_sim_all, cos_sim_prior = None, None

                if cos_sim_prior is not None:
                    tracked_objs[obj_idx]['clip_sim'] = torch.mean(cos_sim_prior).item()
                    tracked_objs[obj_idx]['clip_sim_min'] = torch.min(cos_sim_prior).item()
                    tracked_objs[obj_idx]['clip_sim_max'] = torch.max(cos_sim_prior).item()
                if cos_sim_all is not None:
                    tracked_objs[obj_idx]['clip_sim_a'] = torch.mean(cos_sim_all).item()
                    tracked_objs[obj_idx]['clip_sim_a_min'] = torch.min(cos_sim_all).item()
                    tracked_objs[obj_idx]['clip_sim_a_max'] = torch.max(cos_sim_all).item()
        else:
            print(f"Skip {anno_fname} as no candidate objects found")

        with open(out_path, 'w') as f:
            json.dump({'tracked_objs': tracked_objs, 'all_tracks': all_tracks}, f)
