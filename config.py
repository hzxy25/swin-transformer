# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import torch
import yaml
from yacs.config import CfgNode as CN

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

_C = CN()

# Base config files
_C.BASE = ['']
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Swin Transformer MoE parameters
_C.MODEL.SWIN_MOE = CN()
_C.MODEL.SWIN_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_MOE.IN_CHANS = 3
_C.MODEL.SWIN_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_MOE.MLP_RATIO = 4.
_C.MODEL.SWIN_MOE.QKV_BIAS = True
_C.MODEL.SWIN_MOE.QK_SCALE = None
_C.MODEL.SWIN_MOE.APE = False
_C.MODEL.SWIN_MOE.PATCH_NORM = True
_C.MODEL.SWIN_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_MOE.TOP_VALUE = 1
_C.MODEL.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_MOE.COSINE_ROUTER = False
_C.MODEL.SWIN_MOE.NORMALIZE_GATE = False
_C.MODEL.SWIN_MOE.USE_BPR = True
_C.MODEL.SWIN_MOE.IS_GSHARD_LOSS = False
_C.MODEL.SWIN_MOE.GATE_NOISE = 1.0
_C.MODEL.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.MODEL.SWIN_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# [SimMIM] Norm target during training
_C.MODEL.SIMMIM = CN()
_C.MODEL.SIMMIM.NORM_TARGET = CN()
_C.MODEL.SIMMIM.NORM_TARGET.ENABLE = False
_C.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = 47

# -----------------------------------------------------------------------------
# Loss settings
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# [SwinFCLoss] 各阶段分支权重
# 用于class SwinFCLoss(nn.Module)的branch_rate参数
_C.LOSS.BRANCH_RATE = [0.4, 0.6, 0.8]
# [SwinFCLoss] alpha参数
_C.LOSS.ALPHA = 0.4

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False
"""
这个配置文件使用了 **YACS (Yet Another Configuration System)** 来管理 Swin Transformer 模型及其训练过程的超参数。以下是 `_C` 配置中各参数的详细解释，按模块分类：

---

### **1. 基础配置 (`_C.BASE`)**
- **`BASE`**: 用于指定继承的基础配置文件路径（YAML 格式），支持多文件继承。  
  示例：`_C.BASE = ['path/to/base_config.yaml']`  
  **作用**：避免重复定义通用配置，支持模块化配置管理。

---

### **2. 数据配置 (`_C.DATA`)**
控制数据加载和预处理：
- **`BATCH_SIZE`**: 单 GPU 的批大小（可被命令行覆盖）。
- **`DATA_PATH`**: 数据集路径。
- **`DATASET`**: 数据集名称（如 `imagenet`）。
- **`IMG_SIZE`**: 输入图像尺寸（默认 `224x224`）。
- **`INTERPOLATION`**: 图像缩放插值方法（`bicubic`/`bilinear`/`random`）。
- **`ZIP_MODE`**: 是否使用压缩包格式数据集（替代文件夹格式）。
- **`CACHE_MODE`**: 数据缓存策略（`part`/`full`）。
- **`PIN_MEMORY`**: 是否锁页内存（加速 GPU 数据传输）。
- **`NUM_WORKERS`**: 数据加载的线程数。
- **`MASK_PATCH_SIZE`**: [SimMIM] 掩码生成器的块大小（自监督学习）。
- **`MASK_RATIO`**: [SimMIM] 掩码比例（如 `0.6` 表示遮挡 60% 的图像块）。

---

### **3. 模型配置 (`_C.MODEL`)**
定义模型结构和超参数：
#### **通用参数**
- **`TYPE`**: 模型类型（如 `swin`）。
- **`NAME`**: 模型名称（如 `swin_tiny_patch4_window7_224`）。
- **`PRETRAINED`**: 预训练权重路径。
- **`RESUME`**: 训练中断后恢复的检查点路径。
- **`NUM_CLASSES`**: 分类类别数（默认 1000）。
- **`DROP_RATE`**: 全连接层 Dropout 率。
- **`DROP_PATH_RATE`**: 随机深度（Stochastic Depth）的丢弃率。
- **`LABEL_SMOOTHING`**: 标签平滑系数（防止过拟合）。

#### **Swin Transformer 专用参数 (`_C.MODEL.SWIN`)**
- **`PATCH_SIZE`**: 图像分块大小（默认 `4x4`）。
- **`IN_CHANS`**: 输入通道数（RGB 为 `3`）。
- **`EMBED_DIM`**: 嵌入维度（初始特征维度）。
- **`DEPTHS`**: 各阶段的 Transformer 层数（如 `[2, 2, 6, 2]`）。
- **`NUM_HEADS`**: 各阶段的多头注意力头数。
- **`WINDOW_SIZE`**: 局部窗口大小（自注意力计算范围）。
- **`MLP_RATIO`**: MLP 扩展比率。
- **`APE`**: 是否使用绝对位置编码（`False` 表示使用相对位置编码）。

#### **其他变体参数**
- **`SWINV2`/`SWIN_MOE`/`SWIN_MLP`**: Swin Transformer V2、混合专家（MoE）和 MLP 变体的专用参数。
- **`SIMMIM`**: 自监督学习（SimMIM）的归一化目标配置。

---

### **4. 训练配置 (`_C.TRAIN`)**
控制训练过程：
- **`EPOCHS`**: 总训练轮次。
- **`WARMUP_EPOCHS`**: 学习率预热轮次。
- **`BASE_LR`**: 基础学习率。
- **`WEIGHT_DECAY`**: 权重衰减（L2 正则化系数）。
- **`CLIP_GRAD`**: 梯度裁剪阈值。
- **`AUTO_RESUME`**: 是否自动恢复最近一次的检查点。
- **`USE_CHECKPOINT`**: 是否使用梯度检查点（节省显存）。

#### **学习率调度器 (`_C.TRAIN.LR_SCHEDULER`)**
- **`NAME`**: 调度器类型（如 `cosine`/`step`）。
- **`DECAY_EPOCHS`**: 学习率衰减间隔（`step` 调度器专用）。
- **`DECAY_RATE`**: 学习率衰减比例。

#### **优化器 (`_C.TRAIN.OPTIMIZER`)**
- **`NAME`**: 优化器类型（如 `adamw`）。
- **`BETAS`**: Adam 优化器的动量参数。
- **`MOMENTUM`**: SGD 优化器的动量。

---

### **5. 数据增强 (`_C.AUG`)**
配置图像增强策略：
- **`COLOR_JITTER`**: 颜色抖动强度。
- **`AUTO_AUGMENT`**: 自动增强策略（如 `rand-m9-mstd0.5-inc1`）。
- **`MIXUP`/`CUTMIX`**: MixUp 和 CutMix 的混合系数。
- **`REPROB`**: 随机擦除概率。

---

### **6. 测试配置 (`_C.TEST`)**
- **`CROP`**: 是否中心裁剪测试图像。
- **`SEQUENTIAL`**: 是否使用顺序采样器。
- **`SHUFFLE`**: 是否打乱测试数据。

---

### **7. 其他配置 (`_C.MISC`)**
- **`OUTPUT`**: 输出目录路径。
- **`SEED`**: 随机种子。
- **`AMP_ENABLE`**: 是否启用自动混合精度（AMP）。
- **`LOCAL_RANK`**: 分布式训练的本地 GPU 编号。
- **`FUSED_WINDOW_PROCESS`**: 是否启用融合窗口操作（加速训练）。

---

### **关键函数**
1. **`_update_config_from_file`**  
   从 YAML 文件加载配置并合并到当前配置中，支持继承基础配置（`BASE` 字段）。

2. **`update_config`**  
   根据命令行参数（`args`）动态覆盖配置项（如批大小、学习率）。

3. **`get_config`**  
   返回配置对象的副本，确保默认配置不被污染。

---

### **设计思想**
- **模块化**：通过嵌套的 `CfgNode` 组织参数（如 `DATA`、`MODEL`、`TRAIN`）。
- **灵活性**：支持从文件、命令行、代码多途径修改配置。
- **可复现性**：完整的配置可保存为 YAML 文件，确保实验可复现。

这个配置文件是 Swin Transformer 官方实现的核心部分，涵盖了从数据加载到模型训练的所有关键参数。
"""

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    if _check_args('branch_rate'):
        # 解析逗号分隔的浮点数列表，如 "0.3,0.5,0.7"
        config.LOSS.BRANCH_RATE = [float(x) for x in args.branch_rate.split(',')]
    if _check_args('alpha'):
        config.LOSS.ALPHA = float(args.alpha)
    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        config.LOCAL_RANK = int(os.environ['LOCAL_RANK'])

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
