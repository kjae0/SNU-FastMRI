import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from src import dataset
from src import losses
from src.losses import SSIMLoss, ssim_loss
from src.models.engine import get_engine_cls
from src.models.build_model import build_model
from src.dataset import build_dataset
from src.masker import build_masker

class Trainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.best_val_loss = 1.0
        self.model_path = cfg['exp_dir']
        self.loss_fns = None  # Will be initialized in run_training

    def get_stage_env(self, stage):
        cfg = self.cfg
        device = self.device

        num_epochs = cfg[f'num_epochs{stage}']
        model_dict = build_model(cfg)
        model = model_dict[f'model{stage}'].to(device)
        sens_net = model_dict['sens_net'].to(device)
        
        engine = get_engine_cls(stage)
        
        if stage == 1:
            engine = get_engine_cls(stage)(sens_net=sens_net, model=model)
    
        elif stage == 2:
            # Load the sensitivity network checkpoint from stage 1 if stage 2 has no checkpoint
            state_dict = torch.load(os.path.join(cfg['stage1_exp_dir'], 'best_model.pt'), map_location=device)
            sens_net.load_state_dict(state_dict['sens_net'])
            engine = get_engine_cls(stage)(sens_net=sens_net, model=model)
            
        elif stage == 3:
            # Load both sensitivity network and model from stage 2 checkpoint if stage 3 has no checkpoint
            state_dict = torch.load(os.path.join(cfg['exp_dir2'], 'best_model.pt'), map_location=device)
            sens_net.load_state_dict(state_dict['sens_net'])
            prev_model = model_dict['model2'].to(device)
            prev_model.load_state_dict(state_dict['model'])
            engine = get_engine_cls(stage)(sens_net=sens_net, prev_models=[prev_model], model=model)
            
        else:
            raise ValueError("Invalid stage number")

        dset_dict = build_dataset(cfg, num_epochs)
        train_dl = dset_dict['train_dl']
        val_dl = dset_dict['val_dl']
        augmentor = dset_dict['augmentor']

        masker_dict = build_masker(cfg, num_epochs)
        masker = masker_dict['train_masker']
        val_maskers = masker_dict['val_masker']

        loss_fns = losses.build_loss_fn(cfg, device)

        milestones = cfg.get(f'milestones{stage}') or [i for i in range(0, num_epochs, cfg['step_size'])]
        optimizer = optim.Adam(engine.parameters(), cfg[f'lr{stage}'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=cfg['gamma'])

        print(f"[INFO] Training Stage {stage}")
        print(f"[INFO] Training {cfg['model']} with {cfg['sens_model']} sensitivity model")
        print(f"[INFO] Loss functions: {loss_fns}")
        print(f"[INFO] Scheduler milestones: {milestones}")
        print(f"[INFO] Expected last learning rate: {cfg[f'lr{stage}'] * cfg['gamma'] ** len(milestones)}")

        env = {
            'num_epochs': num_epochs,
            'model': model,
            'engine': engine,
            'train_dl': train_dl,
            'val_dl': val_dl,
            'augmentor': augmentor,
            'masker': masker,
            'val_maskers': val_maskers,
            'loss_fns': loss_fns,
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
        return env

    def run_training(self, stage, ckpt=None):
        env = self.get_stage_env(stage)
        num_epochs = env['num_epochs']
        engine = env['engine']
        train_dl = env['train_dl']
        val_dl = env['val_dl']
        augmentor = env['augmentor']
        masker = env['masker']
        val_maskers = env['val_maskers']
        optimizer = env['optimizer']
        scheduler = env['scheduler']
        self.loss_fns = env['loss_fns']

        if ckpt:
            print(f"[INFO] Loading checkpoint!")
            engine.load_state_dict(ckpt['engine'])
            optimizer.load_state_dict(ckpt['optimizer'])

        self.train_loop(augmentor, engine, train_dl, val_dl, optimizer, scheduler, masker, val_maskers, num_epochs)

    def train_loop(self, augmentor, engine, train_dl, val_dl, optimizer, scheduler, masker, val_maskers, num_epochs):
        cfg = self.cfg
        device = self.device
        best_val_loss = self.best_val_loss

        val_loss_dir = cfg['val_loss_dir']
        file_path = os.path.join(val_loss_dir, "val_loss_log.txt")
        with open(file_path, 'w') as f:
            f.write("epoch, val_loss\n")

        for epoch in range(num_epochs):
            augmentor.set_epoch(epoch)
            print(f'Epoch #{epoch+1:2d} ............... {cfg["net_name"]} ...............')
            if augmentor.aug_on:
                print("[INFO] Augmentation probability: ", augmentor.schedule_p(), augmentor.current_epoch)
            else:
                print("[INFO] Augmentation off")

            train_loss, train_time = self.train_one_epoch(engine, train_dl, optimizer, masker, epoch, num_epochs)
            print(f"{epoch+1} / {num_epochs} - Train loss: {train_loss:.4g}, TrainTime = {train_time:.4f}s")

            val_losses, reconstructions, targets, maximums, val_time = self.validate(engine, val_dl, val_maskers)

            for i, acc in enumerate(val_maskers):
                print(f"{epoch+1} / {num_epochs} - validation SSIM loss for {acc} acceleration: {val_losses[i]:.4g}")
            val_loss = np.mean(val_losses)

            with open(file_path, 'a') as f:
                f.write(f"{epoch}, {val_loss}\n")

            is_new_best = val_loss < self.best_val_loss
            self.best_val_loss = min(self.best_val_loss, val_loss)

            self.save_model(engine, optimizer, epoch + 1, val_loss, is_new_best)

            print(
                f'Epoch = [{epoch+1:4d}/{num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
            )

            scheduler.step()

            if cfg['acc_scheduler']:
                masker.step()

    def train_one_epoch(self, engine, dl, optimizer, masker, epoch, total_epoch):
        cfg = self.cfg
        device = self.device
        engine.train()
        start_epoch = start_iter = time.perf_counter()
        len_loader = len(dl)
        total_loss = 0.

        for iter, data in enumerate(dl):
            mask, kspace, target, maximum, _, _ = data

            assert mask.dim() == 4
            assert kspace.dim() == 5
            assert target.dim() == 4
            assert maximum.dim() == 2

            if masker is not None:
                mask, _, acceleration = masker(kspace.shape)
                mask = mask.bool()
            else:
                acceleration = None

            kspace = kspace * mask + 0.0
            mask = mask.to(device)
            kspace = kspace.to(device)
            target = target.to(device)
            maximum = maximum.to(device)

            output = engine(kspace, mask)[cfg['output_target_key']]

            loss = 0
            for loss_fn in self.loss_fns:
                try:
                    loss += loss_fn(output, target, maximum)
                except:
                    loss += loss_fn(output, target)

            if acceleration and cfg['acceleration_scaler']:
                loss *= (((16 / acceleration) ** 2) / 4)

            if torch.isnan(loss):
                raise ValueError(f"Loss became Nan at epoch {epoch} iteration {iter}. Stopping execution.")

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(engine.parameters(), cfg['clip_norm'])
            optimizer.step()
            total_loss += loss.item()

            if (iter+1) % cfg['report_interval'] == 0:
                print(
                    f'Epoch = [{epoch+1:3d}/{total_epoch:3d}] '
                    f'Iter = [{iter+1:4d}/{len(dl):4d}] '
                    f'Loss = {loss.item():.4g} '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
                start_iter = time.perf_counter()

        total_loss /= len_loader

        return total_loss, time.perf_counter() - start_epoch

    def validate(self, engine, dl, val_maskers):
        cfg = self.cfg
        device = self.device
        engine.eval()
        start = time.perf_counter()
        val_losses = []

        with torch.no_grad():
            for acc, masker in val_maskers.items():
                reconstructions = defaultdict(dict)
                targets = defaultdict(dict)
                maximums = defaultdict(dict)

                for iter, data in enumerate(dl):
                    mask, kspace, target, maximum, fnames, slices = data

                    if masker is not None:
                        mask, _ = masker(kspace.shape)
                        mask = mask.bool()

                    kspace = kspace * mask + 0.0

                    kspace = kspace.to(device)
                    mask = mask.to(device)
                    output = engine(kspace, mask)[cfg['output_target_key']]

                    for i in range(output.shape[0]):
                        reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                        targets[fnames[i]][int(slices[i])] = target[i].numpy()
                        maximums[fnames[i]][int(slices[i])] = maximum[i].numpy()

                    if (iter+1) % cfg['report_interval'] == 0:
                        print(f"{acc} Acceleration Validation: {iter+1}/{len(dl)}", end='\r')

                for fname in reconstructions:
                    reconstructions[fname] = np.stack(
                        [out for _, out in sorted(reconstructions[fname].items())]
                    )
                for fname in targets:
                    targets[fname] = np.stack(
                        [out for _, out in sorted(targets[fname].items())]
                    )
                for fname in maximums:
                    maximums[fname] = np.stack(
                        [out for _, out in sorted(maximums[fname].items())]
                    )

                val_loss = sum([ssim_loss(targets[fname], reconstructions[fname], maximums[fname][0]) for fname in reconstructions]) / len(reconstructions)
                val_losses.append(val_loss)

        val_time = time.perf_counter() - start
        return val_losses, reconstructions, targets, maximums, val_time

    def save_model(self, engine, optimizer, epoch, val_loss, is_new_best):
        cfg = self.cfg
        exp_dir = self.model_path

        torch.save(
            {
                'epoch': epoch,
                'cfg': cfg,
                'sens_net': engine.sens_net.state_dict(),
                'model': engine.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir,
                'engine': engine.state_dict()
            },
            os.path.join(exp_dir, f'{epoch}_model.pt')
        )

        if is_new_best:
            shutil.copyfile(os.path.join(exp_dir, f'{epoch}_model.pt'), os.path.join(exp_dir, 'best_model.pt'))
