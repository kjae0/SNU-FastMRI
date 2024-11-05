import os
import cv2
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import time
from collections import defaultdict

from src import dataset
from src import losses
from src.losses import SSIMLoss, ssim_loss
from src.models.engine import Stage1Engine, Stage2Engine, Stage3Engine
from src.models.build_model import build_model
from src.dataset import build_dataset
from src.masker import build_masker

def train_loop(cfg, 
               augmentor, 
               engine, 
               train_dl, 
               val_dl,
               optimizer, 
               scheduler, 
               loss_fns, 
               device, 
               masker,
               val_maskers,
               model_path):    
        best_val_loss = 1.
        val_loss = []
        
        val_loss_dir = cfg['val_loss_dir']
        num_epochs = cfg['num_epochs']
        val_dir = cfg['val_dir']
        
        file_path = os.path.join(val_loss_dir, "val_loss_log.txt")
        with open(file_path, 'w') as f:
            f.write("epoch, val_loss\n")        

        for epoch in range(num_epochs):
            augmentor.set_epoch(epoch)
            print(f'Epoch #{epoch+1:2d} ............... {cfg['net_name']} ...............')
            if augmentor.aug_on:
                print("[INFO] Augmentation probability: ", augmentor.schedule_p(), augmentor.current_epoch)
            else:
                print("[INFO] Augmentation off")

            train_loss, train_time = train_one_epoch(cfg, epoch, engine, train_dl, optimizer, scheduler, loss_fns, device, masker, num_epochs)
            print(f"{epoch+1} / {num_epochs} - Train loss: {train_loss:.4g}, TrainTime = {train_time:.4f}s")

            val_losses, reconstructions, targets, maximums, val_time = validate(cfg, engine, val_dl, device, val_maskers)
            
            for i, acc in enumerate(val_maskers):
                print(f"{epoch+1} / {num_epochs} - validation SSIM loss for {acc} acceleration: {val_losses[i]:.4g}")
            val_loss = np.mean(val_losses)
        
            with open(file_path, 'a') as f:
                f.write(f"{epoch}, {val_loss}\n")

            is_new_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)

            save_model(cfg, model_path, epoch + 1, engine, optimizer, val_loss, is_new_best)
            print(
                f'Epoch = [{epoch+1:4d}/{num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
            )
            
            if cfg['acc_scheduler']:
                masker.step()

def train_one_epoch(cfg, epoch, model, dl, optimizer, scheduler, loss_fns, device, masker, total_epoch):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(dl)
    total_loss = 0.

    for iter, data in enumerate(dl):
        mask, kspace, target, maximum, _, _ = data
        
        # mask -> (B, 1, H, W)
        # kspace -> (B, 1, H, W, 2)
        # target -> (B, 1, H, W)
        # maximum -> (B, 1)
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

        output = model(kspace, mask)[cfg['output_target_key']]

        loss = 0
        for loss_fn in loss_fns:
            try:
                loss += loss_fn(output, target, maximum)
            except:
                loss += loss_fn(output, target)
                
        if acceleration and cfg['acceleration_scaler']:
            loss *= (((16 / acceleration) ** 2) / 4)
            
        # if loss became Nan, stop training
        if torch.isnan(loss):
            raise ValueError(f"Loss became Nan at epoch {epoch} iteration {iter}. Stopping execution.")
        
        optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_norm'])
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

    total_loss = total_loss / len_loader
    scheduler.step()

    return total_loss, time.perf_counter() - start_epoch

def validate(args, model, dl, device, val_maskers):
    model.eval()
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
                output = model(kspace, mask)[args.output_target_key]

                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                    targets[fnames[i]][int(slices[i])] = target[i].numpy()
                    maximums[fnames[i]][int(slices[i])] = maximum[i].numpy()
                
                if (iter+1) % args.report_interval == 0:
                    print(f"{acc} Acceleration Validation: {iter+1}/{len(dl)}", end='\r')
                    # break

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
 
    return val_losses, reconstructions, targets, maximums, time.perf_counter() - start

def save_model(cfg, exp_dir, epoch, model, optimizer, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'cfg': cfg,
            'sens_net': model.sens_net.state_dict(),
            'model': model.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir,
            'engine': model.state_dict()
        },
        f=os.path.join(exp_dir, f'{epoch}_model.pt')
    )
    
    if is_new_best:
        shutil.copyfile(os.path.join(exp_dir, f'{epoch}_model.pt'), os.path.join(exp_dir, 'best_model.pt'))
     
def train_1stage(cfg, device, ckpt=None):
    model_dict = build_model(cfg)
    model = model_dict['model1'].to(device)
    sens_net = model_dict['sens_net'].to(device)
    engine = Stage1Engine(sens_net=sens_net, model=model)
    
    dset_dict = build_dataset(cfg, cfg['num_epochs1'])
    train_dl = dset_dict['train_dl']
    val_dl = dset_dict['val_dl']
    augmentor = dset_dict['augmentor']
    
    masker_dict = build_masker(cfg, cfg['num_epochs1'])
    masker = masker_dict['train_masker']
    val_maskers = masker_dict['val_masker']

    loss_fns = losses.build_loss_fn(cfg, device)
    print(f"[INFO] Loss functions: {loss_fns}")
    
    if cfg['milestones1']:
        milestones = cfg['milestones1']
    else:
        milestones = [i for i in range(0, cfg['num_epochs'], cfg['step_size'])]
    
    optimizer = optim.Adam(engine.parameters(), cfg['lr1'])
    
    if ckpt:
        print(f"[INFO] Loading checkpoint!")
        engine.load_state_dict(ckpt['engine'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones, 
                                               gamma=cfg['gamma'])
    
    print("[INFO] Training Stage 1")
    print(f"[INFO] Training {cfg['model']} with {cfg['sens_model']} sensitivity model")
    print(f"[INFO] Scheduler milestones: {milestones}")
    print(f"[INFO] Expected last learning rate: {cfg['lr1'] * cfg['gamma'] ** len(milestones)}")

    train_loop(cfg=cfg,
               augmentor=augmentor,
               engine=engine,
               train_dl=train_dl,
               val_dl=val_dl,
               optimizer=optimizer,
               scheduler=scheduler,
               loss_fns=loss_fns,
               device=device,
               masker=masker,
               val_maskers=val_maskers,
               model_path=cfg['exp_dir'])

def train_2stage(cfg, device, ckpt=None):
    model_dict = build_model(cfg)
    model = model_dict['model2'].to(device)
    sens_net = model_dict['sens_net'].to(device)
    
    if ckpt == None:
        state_dict = torch.load(os.path.join(cfg['stage1_exp_dir'], 'best_model.pt'), map_location=device)['sens_net']
        sens_net.load_state_dict(state_dict)
    else:
        print(f"[INFO] Loading checkpoint!")
        sens_net.load_state_dict(ckpt['sens_net'])
        
    engine = Stage2Engine(sens_net=sens_net, model=model)
    
    dset_dict = build_dataset(cfg, cfg['num_epochs2'])
    train_dl = dset_dict['train_dl']
    val_dl = dset_dict['val_dl']
    augmentor = dset_dict['augmentor']
    
    masker_dict = build_masker(cfg, cfg['num_epochs2'])
    masker = masker_dict['train_masker']
    val_maskers = masker_dict['val_masker']

    loss_fns = losses.build_loss_fn(cfg, device)
    print(f"[INFO] Loss functions: {loss_fns}")
    
    if cfg['milestones2']:
        milestones = cfg['milestones2']
    else:
        milestones = [i for i in range(0, cfg['num_epochs'], cfg['step_size'])]
    optimizer = optim.Adam(engine.parameters(), cfg['lr2'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones, 
                                               gamma=cfg['gamma'])
    
    print("[INFO] Training Stage 2")
    print(f"[INFO] Training {cfg['model']} with {cfg['sens_model']} sensitivity model")
    print(f"[INFO] Scheduler milestones: {milestones}")
    print(f"[INFO] Expected last learning rate: {cfg['lr2'] * cfg['gamma'] ** len(milestones)}")
    
    train_loop(cfg=cfg,
               augmentor=augmentor,
               engine=engine,
               train_dl=train_dl,
               val_dl=val_dl,
               optimizer=optimizer,
               scheduler=scheduler,
               loss_fns=loss_fns,
               device=device,
               masker=masker,
               val_maskers=val_maskers,
               model_path=cfg['exp_dir'])

def train_3stage(cfg, device, ckpt=None):
    model_dict = build_model(cfg)
    prev_model = model_dict['model2'].to(device)
    model = model_dict['model3'].to(device)
    sens_net = model_dict['sens_net'].to(device)
    
    state_dict = torch.load(os.path.join(cfg['exp_dir2'], 'best_model.pt'), map_location=device)
    sens_net.load_state_dict(state_dict['sens_net'])
    prev_model.load_state_dict(state_dict['model'])
    engine = Stage3Engine(sens_net=sens_net, prev_models=[prev_model], model=model)
        
    dset_dict = build_dataset(cfg, cfg['num_epochs3'])
    train_dl = dset_dict['train_dl']
    val_dl = dset_dict['val_dl']
    augmentor = dset_dict['augmentor']
    
    masker_dict = build_masker(cfg, cfg['num_epochs3'])
    masker = masker_dict['train_masker']
    val_maskers = masker_dict['val_masker']

    loss_fns = losses.build_loss_fn(cfg, device)
    print(f"[INFO] Loss functions: {loss_fns}")
    
    if cfg['milestones3']:
        milestones = cfg['milestones3']
    else:
        milestones = [i for i in range(0, cfg['num_epochs'], cfg['step_size'])]
    optimizer = optim.Adam(engine.parameters(), cfg['lr3'])
    
    if ckpt:
        print(f"[INFO] Loading checkpoint!")
        engine.load_state_dict(ckpt['engine'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones, 
                                               gamma=cfg['gamma'])
    
    print("[INFO] Training Stage 3")
    print(f"[INFO] Training {cfg['model']} with {cfg['sens_model']} sensitivity model")
    print(f"[INFO] Scheduler milestones: {milestones}")
    print(f"[INFO] Expected last learning rate: {cfg['lr3'] * cfg['gamma'] ** len(milestones)}")
    
    train_loop(cfg=cfg,
               augmentor=augmentor,
               engine=engine,
               train_dl=train_dl,
               val_dl=val_dl,
               optimizer=optimizer,
               scheduler=scheduler,
               loss_fns=loss_fns,
               device=device,
               masker=masker,
               val_maskers=val_maskers,
               model_path=cfg['exp_dir'])
    