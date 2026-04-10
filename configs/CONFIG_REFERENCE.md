# TubeletGraph Configuration Reference

## Pipeline Overview

TubeletGraph runs in stages:

1. **Entity Segmentation** — CropFormer generates per-frame region proposals (superpixels)
2. **Tubelet Construction** — SAM2 tracks those regions across frames, filling in new ones over time
3. **Semantic Similarity** — FC-CLIP features score each tubelet's similarity to the prompt object
4. **Prediction** — tubelets passing metric thresholds are merged into the final mask
5. **VLM** — a vision-language model describes transformations
6. **Evaluation** — IoU, temporal localization, semantic accuracy

---

## `paths`

| Parameter | Description |
|-----------|-------------|
| `intermdir` | Directory for all intermediate artifacts: entity JSONs from CropFormer, tubelet JSONs from SAM2 tracking, FC-CLIP enriched tubelets, cached proximity masks. |
| `outdir` | Final per-video prediction JSONs (from `get_prediction.py`), consumed by VLM and evaluation scripts. |
| `visdir` | Visualization outputs: tubelet MP4 videos, prediction overlays, PDF summaries. |
| `evaldir` | Quantitative evaluation results: IoU tables, tracking metrics, temporal localization precision/recall, semantic accuracy scores. |

---

## `datasets.*`

| Parameter | Description |
|-----------|-------------|
| `name` | Dataset identifier, used in directory naming (e.g., `entities_vost_cropformer`). |
| `data_dir` | Root directory of the dataset. |
| `image_dir` | Path to `JPEGImages/` containing per-video frame directories. |
| `anno_dir` | Path to `Annotations/` with ground-truth masks (used in evaluation). |
| `split_dir` | Directory containing `.txt` files listing video names per split (e.g., `val.txt`). |
| `image_format` | Glob pattern for frame files (e.g., `frame*.jpg`). Varies by dataset naming convention. |
| `anno_format` | Glob pattern for annotation files (e.g., `frame*.png`). |
| `fps` | Assumed frames-per-second for the dataset. Used for: (1) writing visualization videos at the correct playback rate (`vis/tubelets.py`), and (2) converting frame-index gaps to "X seconds ago" in VLM prompts (`vlm/prompt_vlm.py`). Does **not** subsample frames during processing. |

---

## `entity_seg.cropformer`

Configures the CropFormer entity segmentation model (pipeline step 1), used in `TubeletGraph/entity_segmentation/cropformer.py`.

| Parameter | Description |
|-----------|-------------|
| `project_path` | Path to the CropFormer project directory (under detectron2). Used to set `sys.path` so CropFormer imports resolve. |
| `config_path` | Detectron2 YAML config for the CropFormer model architecture (HorNet backbone, Mask2Former decoder). |
| `opts` | Detectron2 config overrides as a flat key-value list. Sets `MODEL.WEIGHTS` to the pretrained checkpoint. |
| `confidence_threshold` | Minimum instance confidence score (0–1) for a CropFormer proposal to be kept. After inference, masks with `score < threshold` are discarded. Lower values keep more (noisier) proposals; higher values keep fewer but more confident regions. |

---

## `entity_seg.sam_automask`

Defined in config but **not currently wired** into the pipeline Python code. Parameters correspond to `SAM2AutomaticMaskGenerator` from the vendored SAM2 library, reserved for an alternative entity segmentation method.

| Parameter | Description |
|-----------|-------------|
| `model_weights` | Path to SAM2 model checkpoint. |
| `config` | SAM2 model architecture config. |
| `points_per_side` | Grid density for prompt points. Total points = `points_per_side²`. More points = finer coverage but slower. |
| `points_per_batch` | How many points to process simultaneously on the GPU. Higher = faster but more VRAM. |
| `pred_iou_thresh` | Predicted IoU quality filter (0–1). Masks with predicted IoU below this are discarded. |
| `stability_score_thresh` | Stability score filter (0–1). Measures how stable the mask is under small perturbations of the binarization threshold. |
| `stability_score_offset` | Offset for the stability score calculation. The mask is re-binarized at `threshold ± offset` and compared. |
| `mask_threshold` | Logit threshold for binarizing mask predictions. 0.0 = raw sigmoid midpoint. |

---

## `tubelet`

Controls tubelet construction in `TubeletGraph/tubelet/compute_tubelets_sam.py` — the core of the method.

| Parameter | Description |
|-----------|-------------|
| `entity_method` | Label selecting which entity source to load. Determines the intermediate directory name: `entities_{dataset}_{entity_method}`. Must match the suffix used by the entity segmentation script (default: `cropformer`). |

### `tubelet.tracker` (Main Tracker)

The primary SAM2 instance used for all tubelet propagation (initial tracks, conflict resolution, fill tracking).

| Parameter | Description |
|-----------|-------------|
| `name` | Tracker identifier. Must be `SAM2.1` (asserted in code). |
| `module` | Python class to instantiate (`SAM2` in `TubeletGraph/tracker/sam2.py`). |
| `model_weights` | SAM2 checkpoint path. |
| `model_cfg` | SAM2 architecture config. |
| `multi_mask` | **`False`** for the main tracker. Only the single best mask per object per frame is needed for tubelet propagation. |

### Tubelet Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `init_entity_thrd` | 0.0005 | Minimum area fraction (relative to frame area) for a frame-0 entity to be initialized as a SAM2 tracked object. Entities smaller than `H × W × 0.0005` pixels are skipped as noise. Must be ≤ `fill_entity_thrd`. |
| `fill_entity_thrd` | 0.0016 | Minimum area fraction for two purposes: (1) after conflict resolution, tracked objects below this fraction are pruned as too small; (2) when considering new entity proposals mid-video, only proposals larger than this are candidates. Value ≈ `(1/25)²`, roughly a 1/25th-of-frame-side square. |
| `fill_coverage_thrd` | 0.25 | Maximum overlap (0–1) with existing tracks for a new entity to be added. "Coverage" = fraction of the entity mask area already covered by the union of all current track masks. Only entities with coverage < 0.25 are added, ensuring only genuinely new, uncovered regions are filled. Also used during prompt conflict resolution: initial tracks with coverage > 0.25 by the prompt mask are flagged as conflicting. |
| `rm_init_entity_thrd` | 0.75 | Removal threshold for initial entities conflicting with the prompt mask. For flagged tracks (coverage > `fill_coverage_thrd`): if coverage < 0.75, the overlapping portion is erased and the entity is re-tracked (partial trim). If coverage ≥ 0.75, the entity is deleted entirely (mostly the prompt object). |
| `collect_spacing` | 30 | Collection interval in frames. Every N frames, the pipeline scans the preceding window for uncovered entities, adds them with short forward propagation, then re-seeds all newly collected objects and propagates to the end of the video. Smaller = discovers new objects faster; larger = less compute. |

### `tubelet.prox_tracker` (Proximity Tracker)

A **separate** SAM2 instance used to generate multi-mask hypotheses from the prompt object only, providing proximity signals for newly added tubelets.

| Parameter | Description |
|-----------|-------------|
| `name` | Must be `SAM2.1` (asserted). |
| `module` | `SAM2` class. |
| `model_weights` | SAM2 checkpoint (same model, separate instance). |
| `model_cfg` | SAM2 architecture config. |
| `multi_mask` | **`True`** — the critical difference. Enables SAM2's multi-mask output: for each frame, SAM2 returns 3 ranked mask hypotheses (sorted by predicted IoU) instead of 1. Ranks 1 and 2 (non-primary) are used to compute `mm_iou` and `mm_cover` for newly added tubelets. These alternative masks capture "what else SAM2 thinks the prompt object could look like," serving as a **proximity signal**. |

---

## `sem_sim`

Configures FC-CLIP semantic feature extraction in `TubeletGraph/semantic_sim/compute_sim_fcclip.py`.

| Parameter | Description |
|-----------|-------------|
| `name` | Identifier for the semantic similarity method (used in directory naming). |
| `project_path` | Path to FC-CLIP project directory (for `sys.path`). |
| `config_path` | Detectron2 config for FC-CLIP (ConvNeXt-Large backbone, panoptic segmentation head). |
| `opts` | Config overrides setting the model weights checkpoint. |

---

## `methods.Ours`

Configures the final prediction step in `TubeletGraph/tracker/ours.py`.

| Parameter | Description |
|-----------|-------------|
| `name` | Method display name. |
| `module` | Python class name (`TubeletGraph`), instantiated via `get_tracker`. |
| `tubelet_dirname` | Format string for the directory containing FC-CLIP-enriched tubelet JSONs. `{}` is replaced with the dataset name. E.g., `tubelets_{}_cropformer_fcclip` → `intermdir/tubelets_vost_cropformer_fcclip/`. The `_fcclip` suffix means it reads the output of the semantic similarity step. |

### `methods.Ours.thrds` — Tubelet Selection Thresholds

These two thresholds are applied **conjunctively** — a tubelet must pass **all** of them to be included in the final merged prediction mask. Only tubelets added after frame 0 are filtered; the prompt object is always included.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mm_cover` | 0.3 | **Multi-mask coverage threshold.** For each candidate tubelet, `mm_cover` is the maximum fraction of the tubelet's entity mask covered by SAM2's alternative (rank 1 or 2) mask hypotheses for the prompt object. Keep only tubelets with `mm_cover > 0.3`. This is a **geometric proximity filter**: "does SAM2 think this region could be part of the prompt object?" |
| `clip_sim_max` | 0.7 | **Maximum CLIP cosine similarity threshold.** FC-CLIP features of the tubelet (on frames after it appears) are compared to the prompt object's features (on frames before the tubelet appears). `clip_sim_max` is the maximum of those cosine similarities. Keep only tubelets with `clip_sim_max > 0.7`. This is a **semantic appearance filter**: "does this region look like the prompt object in CLIP feature space?" |

---

## `vlm`

Configures the vision-language model step in `TubeletGraph/vlm/prompt_vlm.py`, which describes what transformation happens to the object.

| Parameter | Description |
|-----------|-------------|
| `model_name` | API model identifier. `gpt-4.1` for OpenAI, or `Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic` for HuggingFace router. Determines which API client is used and the output directory suffix. |
| `init_color_rgb` | RGB color `[245, 0, 150]` (pink) used to overlay the **prompt object's mask** in images sent to the VLM. |
| `init_color_name` | Human-readable name for the init color, used in text prompts to the VLM (e.g., "the object highlighted in **pink**"). |
| `query_color_rgb` | RGB color `[0, 245, 245]` (cyan) used to overlay **newly emerged tubelet masks** in images sent to the VLM. |
| `query_color_name` | Human-readable name for the query color, used in VLM text prompts (e.g., "the new region shown in **cyan-blue**"). |

---

## `eval`

Configures evaluation scripts under `eval/`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obj_id` | `'0'` | Which object ID key to read from prediction JSONs when evaluating. The merged prediction is stored under key `'0'`, so this matches the standard pipeline output. |
| `skip_first_frame` | `True` | If true, frame 0 is excluded from IoU/precision/recall computation. Since frame 0's mask is the given prompt (trivially correct), skipping it gives a more meaningful evaluation of tracking quality. |
| `temploc_max_pad` | 0 | Temporal localization padding in seconds. Controls the tolerance window for matching predicted transformation timestamps to ground-truth time ranges. With `0`, only exact matches are evaluated. Higher values add ±N seconds of tolerance, generating precision/recall curves at multiple strictness levels. Internally each unit is multiplied by 30 frames. |
