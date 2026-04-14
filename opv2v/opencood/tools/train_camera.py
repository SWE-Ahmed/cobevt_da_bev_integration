import argparse
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
import copy
import numpy as np
from itertools import cycle
from opencood.utils.da_bev_utils import compute_da_bev_qal_loss
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--hypes_yaml",
        type=str,
        required=True,
        help="data generation yaml file needed ",
    )
    parser.add_argument("--model_dir", default="", help="Continued training path")
    parser.add_argument(
        "--half", action="store_true", help="whether train with half precision"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--seed", default=0, type=int, help="seed for training")
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print("-----------------Seed Setting----------------------")
    seed = train_utils.init_random_seed(None if opt.seed == 0 else opt.seed)
    hypes["train_params"]["seed"] = seed
    print("Set seed to %d" % seed)
    train_utils.set_random_seed(seed)

    print("-----------------Dataset Building------------------")

    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_val_dataset = build_dataset(
        hypes, visualize=False, train=True, validate=True
    )

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_val_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes["train_params"]["batch_size"], drop_last=True
        )

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch,
        )
        val_loader = DataLoader(
            opencood_val_dataset,
            sampler=sampler_val,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            opencood_val_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
        )

    print("---------------Building Target Dataloader (Adver-City)------------------")
    target_hypes = copy.deepcopy(hypes)
    # Ensure this points to your Adver-City training directory in your yaml
    target_hypes["root_dir"] = hypes["uda_config"]["target_data_dir"]

    opencood_target_dataset = build_dataset(target_hypes, visualize=False, train=True)

    if opt.distributed:
        sampler_target = DistributedSampler(opencood_target_dataset)
        batch_sampler_target = torch.utils.data.BatchSampler(
            sampler_target, hypes["train_params"]["batch_size"], drop_last=True
        )
        target_loader = DataLoader(
            opencood_target_dataset,
            batch_sampler=batch_sampler_target,
            num_workers=8,
            collate_fn=opencood_target_dataset.collate_batch,
        )
    else:
        target_loader = DataLoader(
            opencood_target_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=8,
            collate_fn=opencood_target_dataset.collate_batch,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )

    print("---------------Creating Model------------------")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    model.to(device)
    model_without_ddp = model

    # --- DA-BEV MODIFICATION START ---
    # The default train_utils.load_saved_model enforces strict=True.
    # If it crashes loading the discriminators, you will need to edit train_utils.py
    # to set strict=False, or just ignore the error if it loads them as uninitialized.

    print("Freezing early backbone layers to protect source domain performance...")
    for name, param in model_without_ddp.named_parameters():
        # Freeze early ResNet/EfficientNet layers. Adjust names based on your architecture.
        if "backbone" in name and (
            "layer1" in name or "layer2" in name or "stem" in name
        ):
            param.requires_grad = False
    # --- DA-BEV MODIFICATION END ---

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    trainable_params = filter(lambda p: p.requires_grad, model_without_ddp.parameters())
    # Note: setup_optimizer usually expects a model, so if this crashes, you may need to
    # bypass setup_optimizer and initialize it directly:
    # optimizer = torch.optim.Adam(trainable_params, lr=hypes['train_params']['lr'])
    try:
        optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    except:
        optimizer = torch.optim.Adam(trainable_params, lr=hypes["train_params"]["lr"])

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    # lr scheduler setup
    epoches = hypes["train_params"]["epoches"]
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    print("Training start with num steps of %d" % num_steps)
    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print("learning rate %.7f" % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, (source_batch, target_batch) in enumerate(
            zip(train_loader, cycle(target_loader))
        ):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            source_batch = train_utils.to_device(source_batch, device)
            target_batch = train_utils.to_device(target_batch, device)

            # --- DA-BEV Alpha Schedule ---
            current_step = epoch * num_steps + i
            max_warmup = hypes["uda_config"]["warmup_steps"]
            p = min(1.0, float(current_step) / max_warmup)
            alpha = (2.0 / (1.0 + np.exp(-10 * p)) - 1) * hypes["uda_config"][
                "grl_alpha_max"
            ]

            if not opt.half:
                # Source Forward (Segmentation + Domain)
                src_output = model(source_batch["ego"], alpha=alpha, is_target=False)
                # Target Forward (Domain Only)
                tgt_output = model(target_batch["ego"], alpha=alpha, is_target=True)

                # Source Supervised Loss
                seg_loss = criterion(src_output, source_batch["ego"])

                # Adversarial Loss
                qal_loss = compute_da_bev_qal_loss(
                    src_iv_pred=src_output["domain_pred_iv"],
                    tgt_iv_pred=tgt_output["domain_pred_iv"],
                    src_bev_pred=src_output["domain_pred_bev"],
                    tgt_bev_pred=tgt_output["domain_pred_bev"],
                )

                final_loss = seg_loss + (
                    hypes["uda_config"]["domain_loss_weight"] * qal_loss
                )
            else:
                with torch.cuda.amp.autocast():
                    src_output = model(
                        source_batch["ego"], alpha=alpha, is_target=False
                    )
                    tgt_output = model(target_batch["ego"], alpha=alpha, is_target=True)
                    seg_loss = criterion(src_output, source_batch["ego"])
                    qal_loss = compute_da_bev_qal_loss(
                        src_output["domain_pred_iv"],
                        tgt_output["domain_pred_iv"],
                        src_output["domain_pred_bev"],
                        tgt_output["domain_pred_bev"],
                    )
                    final_loss = seg_loss + (
                        hypes["uda_config"]["domain_loss_weight"] * qal_loss
                    )

            # Update progress bar description with loss details
            pbar2.set_description(
                f"Loss: {final_loss.item():.3f} | Seg: {seg_loss.item():.3f} | QAL: {qal_loss.item():.3f}"
            )

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            # Log QAL explicitly to tensorboard
            writer.add_scalar("Loss/Seg_Loss", seg_loss.item(), current_step)
            writer.add_scalar("Loss/QAL_Loss", qal_loss.item(), current_step)
            writer.add_scalar("UDA/Alpha", alpha, current_step)

            for lr_idx, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar("lr_%d" % lr_idx, param_group["lr"], current_step)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            scheduler.step_update(current_step)

        if epoch % hypes["train_params"]["eval_freq"] == 0:
            valid_ave_loss = []
            dynamic_ave_iou = []
            static_ave_iou = []
            lane_ave_iou = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data["ego"])

                    final_loss = criterion(output_dict, batch_data["ego"])
                    valid_ave_loss.append(final_loss.item())

                    # visualization purpose
                    output_dict = opencood_val_dataset.post_process(
                        batch_data["ego"], output_dict
                    )
                    train_utils.save_bev_seg_binary(
                        output_dict, batch_data, saved_path, i, epoch
                    )
                    iou_dynamic, iou_static = cal_iou_training(batch_data, output_dict)
                    static_ave_iou.append(iou_static[1])
                    dynamic_ave_iou.append(iou_dynamic[1])
                    lane_ave_iou.append(iou_static[2])

            valid_ave_loss = statistics.mean(valid_ave_loss)
            static_ave_iou = statistics.mean(static_ave_iou)
            lane_ave_iou = statistics.mean(lane_ave_iou)
            dynamic_ave_iou = statistics.mean(dynamic_ave_iou)

            print(
                "At epoch %d, the validation loss is %f,"
                "the dynamic iou is %f, t"
                "he road iou is %f"
                "the lane ious is %f"
                % (epoch, valid_ave_loss, dynamic_ave_iou, static_ave_iou, lane_ave_iou)
            )

            writer.add_scalar("Validate_Loss", valid_ave_loss, epoch)
            writer.add_scalar("Dynamic_Iou", dynamic_ave_iou, epoch)
            writer.add_scalar("Road_IoU", static_ave_iou, epoch)
            writer.add_scalar("Lane_IoU", static_ave_iou, epoch)

        if epoch % hypes["train_params"]["save_freq"] == 0:
            torch.save(
                model_without_ddp.state_dict(),
                os.path.join(saved_path, "net_epoch%d.pth" % (epoch + 1)),
            )

        opencood_train_dataset.reinitialize()


if __name__ == "__main__":
    main()