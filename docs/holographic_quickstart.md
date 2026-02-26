# Quick Start: Holographic Imaging in MMDetection

## Installation

No additional dependencies needed beyond standard MMDetection requirements.

## 1. Prepare Your Data

Organize your holographic dataset:
```
data/holographic/
├── annotations/
│   ├── train.json      # COCO format annotations
│   └── val.json
├── train/
│   ├── holo_001.png    # Holographic intensity images
│   ├── holo_002.png
│   └── ...
├── train_depth/        # Optional: depth maps
│   ├── holo_001.png
│   ├── holo_002.png
│   └── ...
└── val/
    └── ...
```

## 2. Create a Configuration File

```python
# configs/faster_rcnn/faster_rcnn_r50_fpn_holographic.py

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/holographic_detection.py',  # Use holographic config
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Customize if needed
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)
```

## 3. Train Your Model

```bash
# Single GPU
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_holographic.py

# Multiple GPUs
bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_holographic.py 8
```

## 4. Test Your Model

```bash
# Test
python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_holographic.py \
    work_dirs/faster_rcnn_r50_fpn_holographic/latest.pth \
    --eval bbox

# Inference on images
python demo/image_demo.py \
    data/holographic/test/holo_001.png \
    configs/faster_rcnn/faster_rcnn_r50_fpn_holographic.py \
    work_dirs/faster_rcnn_r50_fpn_holographic/latest.pth
```

## Advanced Usage

### Enable Depth Map Loading

```python
train_pipeline = [
    dict(
        type='LoadHolographicImage',
        load_depth=True,      # Enable depth loading
        load_phase=False,
        to_float32=False
    ),
    # ... rest of pipeline
]
```

### Custom Classes

```python
# In your config file
data = dict(
    train=dict(
        type='HolographicDataset',
        classes=('cell', 'particle', 'debris'),  # Your custom classes
        # ... other configs
    )
)
```

### Using with Different Backbones

Just change the base model config:
```python
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',  # Use Mask R-CNN instead
    '../_base_/datasets/holographic_detection.py',
    # ...
]
```

## API Usage

### Programmatic Dataset Creation

```python
from mmdet.datasets import HolographicDataset

dataset = HolographicDataset(
    ann_file='data/holographic/annotations/train.json',
    img_prefix='data/holographic/train/',
    depth_prefix='data/holographic/train_depth/',
    pipeline=train_pipeline
)
```

### Pipeline Component

```python
from mmdet.datasets.pipelines import LoadHolographicImage

loader = LoadHolographicImage(
    load_depth=True,
    load_phase=True,
    to_float32=False
)

# Use in pipeline
results = loader(results_dict)
```

## Examples

See `docs/holographic_imaging.md` for detailed documentation and examples.

## Troubleshooting

**Q: Depth/phase maps not loading?**
- Ensure files exist at specified prefix paths
- Check filename matches between intensity and depth/phase images
- Loading silently fails if files don't exist (by design)

**Q: How do I visualize holographic data?**
- Use standard MMDetection visualization tools
- Depth/phase maps can be visualized separately as grayscale images

**Q: Can I use this with my existing detector configs?**
- Yes! Just change the dataset config to use HolographicDataset
- No changes needed to model architecture

## Support

For issues or questions:
1. Check the full documentation: `docs/holographic_imaging.md`
2. Review test examples: `tests/test_data/test_datasets/test_holographic_dataset.py`
3. Open an issue on GitHub

## Citation

```bibtex
@misc{mmdet-holographic,
  title={Holographic Imaging Support for MMDetection},
  year={2026}
}
```
