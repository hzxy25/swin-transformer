# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from torch import nn

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
import torch.nn.functional as F




# ==============================================
# 临时测试用硬编码参数 - 后期需删除
TEMP_CFG_PATH = "configs/swin/swinfc_base_patch4_window7_224.yaml"
TEMP_DATA_PATH = r"E:\tool\Data\imagenet-mini"
TEMP_BATCH_SIZE = 4
TEMP_OUTPUT = "./output_test"
TEMP_TAG = "test_run"
TEMP_BRANCH_RATE = [0.4, 0.6, 0.8]  # branch_rate参数
TEMP_ALPHA = 0.4  # alpha参数

# 临时环境变量设置 - 后期需删除
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
# ==============================================



# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])




class SwinFCLoss(nn.Module):
    def __init__(self, num_stages=3, branch_rate=[0.4, 0.6, 0.8], alpha=0.4):
        super().__init__()
        self.num_stages = num_stages  # 分支数量（论文中为最后3个阶段）
        self.beta = branch_rate  # 各阶段权重
        self.alpha = alpha  # 对比损失超参数

    def forward(self, x_list, x_fsm, targets):
        # 1. 骨干网络损失 (L_swin)：x_list[-1] 是骨干输出
        loss_swin = F.cross_entropy(x_list[-1], targets)

        # 2. 各阶段分支损失 (sum β_i * L_stage^i)：x_list[0..num_stages-1] 是分支输出
        loss_stage = 0.0
        for i in range(self.num_stages):
            loss_stage += self.beta[i] * F.cross_entropy(x_list[i], targets)

        # 3. 对比损失 (L_con)：修复mask维度
        batch_size = x_fsm.shape[0]

        # 若targets是独热编码（[B, C]），转换为类别索引（[B]）
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)  # [B, C] -> [B]

        # 计算掩码（此时targets为[B]）
        mask = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()  # [B, B]
        cos_sim = F.cosine_similarity(x_fsm.unsqueeze(1), x_fsm.unsqueeze(0), dim=2)  # [B, B]

        # 计算对比损失
        loss_con = (mask * (1 - cos_sim) + (1 - mask) * F.relu(cos_sim - self.alpha)).mean()

        # 总损失
        total_loss = loss_swin + loss_stage + loss_con
        return total_loss

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)

    # 临时
    parser.add_argument('--cfg', type=str, default=TEMP_CFG_PATH, metavar="FILE", help='path to config file')  # 修改为default
    # parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--branch-rate', type=str,
                        help="Comma-separated list of branch rates for SwinFCLoss, e.g. '0.3,0.5,0.7'")
    parser.add_argument('--alpha', type=float,
                        help="Alpha parameter for SwinFCLoss")
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
    # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # args, unparsed = parser.parse_known_args()
    # 临时测试用修改点2: 如果未提供命令行参数，使用硬编码值
    # ==============================================
    # 临时测试用代码开始 - 后期需删除
    args, unparsed = parser.parse_known_args([])  # 传入空列表模拟无命令行参数

    # 设置硬编码参数值
    args.data_path = TEMP_DATA_PATH
    args.batch_size = TEMP_BATCH_SIZE
    args.output = TEMP_OUTPUT
    args.tag = TEMP_TAG
    args.branch_rate = ",".join(map(str, TEMP_BRANCH_RATE))  # 转换为字符串格式
    args.alpha = TEMP_ALPHA
    # 临时测试用代码结束
    # ==============================================
    config = get_config(args)
    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()
    if world_size > 1:  # 仅在分布式环境下使用 DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK],
            broadcast_buffers=False
        )
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = SwinFCLoss(num_stages=len(config.LOSS.BRANCH_RATE), branch_rate=config.LOSS.BRANCH_RATE,alpha= config.LOSS.ALPHA)
    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            x_list, x_fsm = model(samples)
        loss = criterion(x_list, x_fsm, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    # 初始化多分类器的指标记录器（数量与x_list长度一致）
    num_outputs = len(config.LOSS.BRANCH_RATE) + 1  # 分支数量 + 主干网络
    loss_meters = [AverageMeter() for _ in range(num_outputs)]
    acc1_meters = [AverageMeter() for _ in range(num_outputs)]
    acc5_meters = [AverageMeter() for _ in range(num_outputs)]
    # 总平均指标记录器
    total_acc1 = AverageMeter()
    total_acc5 = AverageMeter()
    total_loss = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # 模型输出：多阶段分支分类器结果(x_list) + FSM特征(x_fsm)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            x_list, x_fsm = model(images)  # x_list包含各阶段输出

        # 计算每个分类器的损失和精度
        for i in range(num_outputs):
            output = x_list[i]
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # 同步多卡结果
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            # 更新单分类器指标
            loss_meters[i].update(loss.item(), target.size(0))
            acc1_meters[i].update(acc1.item(), target.size(0))
            acc5_meters[i].update(acc5.item(), target.size(0))

        # 计算总平均指标（所有分类器的均值）
        avg_acc1 = sum([m.avg for m in acc1_meters]) / num_outputs
        avg_acc5 = sum([m.avg for m in acc5_meters]) / num_outputs
        avg_loss = sum([m.avg for m in loss_meters]) / num_outputs
        total_acc1.update(avg_acc1, target.size(0))
        total_acc5.update(avg_acc5, target.size(0))
        total_loss.update(avg_loss, target.size(0))

        # 记录时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 打印日志（每PRINT_FREQ步）
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            # 打印各阶段性能
            stage_logs = []
            for i in range(num_outputs):
                stage_logs.append(
                    f'Stage{i+1}: Acc@1 {acc1_meters[i].val:.3f} '
                    f'Loss {loss_meters[i].val:.4f}'
                )
            stage_logs = ' | '.join(stage_logs)
            # 打印总性能
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Avg Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                f'Avg Acc@1 {total_acc1.val:.3f} ({total_acc1.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB\t| {stage_logs}'
            )

    # 最终日志：总平均和各阶段结果
    logger.info('='*50)
    logger.info(f'Final Avg: Acc@1 {total_acc1.avg:.3f} Acc@5 {total_acc5.avg:.3f} Loss {total_loss.avg:.4f}')
    for i in range(num_outputs):
        logger.info(
            f'Stage{i+1} Final: Acc@1 {acc1_meters[i].avg:.3f} '
            f'Acc@5 {acc5_meters[i].avg:.3f} Loss {loss_meters[i].avg:.4f}'
        )
    logger.info('='*50)

    return total_acc1.avg, total_acc5.avg, total_loss.avg

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
