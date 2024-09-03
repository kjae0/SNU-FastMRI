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
from src import utils
from src.loss_function import SSIMLoss, ssim_loss
from src.models.engine import Stage1Engine, Stage2Engine, Stage3Engine
from src.models.build_model import build_model
from src.dataset import build_dataset
from src.masker import build_masker

def train_loop(args, 
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
               model_path,
               stage):    
        best_val_loss = 1.
        val_loss = []
        
        if stage == 1:
            val_loss_dir = args.val_loss_dir1
            num_epochs = args.num_epochs1
            val_dir = args.val_dir1
        elif stage == 2:
            val_loss_dir = args.val_loss_dir2
            num_epochs = args.num_epochs2
            val_dir = args.val_dir2
        elif stage == 3:
            val_loss_dir = args.val_loss_dir3
            num_epochs = args.num_epochs3
            val_dir = args.val_dir3
        else:
            raise ValueError(f"Unknown stage: {stage}. Must be one of [1, 2, 3]")
        
        file_path = os.path.join(val_loss_dir, "train_loss_log.txt")
        with open(file_path, 'w') as f:
            f.write("epoch, train_loss\n")        

        for epoch in range(num_epochs):
            augmentor.set_epoch(epoch)
            print(f'Epoch #{epoch+1:2d} ............... {args.net_name} ...............')
            if augmentor.aug_on:
                print("Augmentation probability: ", augmentor.schedule_p(), augmentor.current_epoch)
            else:
                print("Augmentation off")

            train_loss, train_time = train_one_epoch(args, epoch, engine, train_dl, optimizer, scheduler, loss_fns, device, masker, num_epochs)
            print(f"{epoch+1} / {num_epochs} - validation Train loss: {train_loss:.4g}, TrainTime = {train_time:.4f}s")
        
            with open(file_path, 'a') as f:
                f.write(f"{epoch}, {train_loss}\n")

            # is_new_best = val_loss < best_val_loss
            # best_val_loss = min(best_val_loss, val_loss)

            save_model(args, model_path, epoch + 1, engine, optimizer)
            print(
                f'Epoch = [{epoch+1:4d}/{num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                # f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
            )
            
            if args.acc_scheduler:
                masker.step()

def train_one_epoch(args, epoch, model, dl, optimizer, scheduler, loss_fns, device, masker, total_epoch):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(dl)
    total_loss = 0.

    for iter, data in enumerate(dl):
        mask, kspace, target, maximum, _, _ = data
        
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

        output = model(kspace, mask)[args.output_target_key]

        loss = 0
        for loss_fn in loss_fns:
            try:
                loss += loss_fn(output, target, maximum)
            except:
                loss += loss_fn(output, target)
                
        if acceleration and args.acceleration_scaler:
            loss *= (((16 / acceleration) ** 2) / 4)
        
        optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        total_loss += loss.item()
        
        if (iter+1) % args.report_interval == 0:
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

def save_model(args, exp_dir, epoch, model, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'sens_net': model.sens_net.state_dict(),
            'model': model.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir,
            'engine': model.state_dict()
        },
        f=os.path.join(exp_dir, f'{epoch}_model.pt')
    )
    shutil.copyfile(os.path.join(exp_dir, f'{epoch}_model.pt'), os.path.join(exp_dir, 'best_model.pt'))
     
def train_1stage(args, device, ckpt=None):
    model_dict = build_model(args)
    model = model_dict['model1'].to(device)
    sens_net = model_dict['sens_net'].to(device)
    
    engine = Stage1Engine(sens_net=sens_net, model=model)
    
    dset_dict = build_dataset(args, args.num_epochs1)
    train_dl = dset_dict['train_dl']
    val_dl = dset_dict['val_dl']
    augmentor = dset_dict['augmentor']
    
    masker_dict = build_masker(args, args.num_epochs1)
    masker = masker_dict['train_masker']
    val_maskers = masker_dict['val_masker']

    loss_fns = []
    
    for lf in args.losses:
        if lf == "s":
            loss_fn = SSIMLoss().to(device)
        elif lf == 'l':
            loss_fn = nn.L1Loss().to(device)
        loss_fns.append(loss_fn)
        
    print(f"Loss functions: {loss_fns}")
    
    if args.milestones1:
        milestones = args.milestones1
    else:
        milestones = [i for i in range(0, args.num_epochs, args.step_size)]
    
    optimizer = optim.Adam(engine.parameters(), args.lr1)
    
    if ckpt:
        print(f"Loading checkpoint!")
        engine.load_state_dict(ckpt['engine'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones, 
                                               gamma=args.gamma)
    
    print("Training Stage 1")
    print(f"Training {args.model} with {args.sens_model} sensitivity model")
    print(f"Scheduler milestones: {milestones}")
    print(f"Expected last learning rate: {args.lr1 * args.gamma ** len(milestones)}")

    train_loop(args=args,
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
               model_path=args.exp_dir1,
               stage=1)


def train_2stage(args, device, ckpt=None):
    model_dict = build_model(args)
    model = model_dict['model2'].to(device)
    sens_net = model_dict['sens_net'].to(device)
    
    if ckpt == None:
        state_dict = torch.load(os.path.join(args.exp_dir1, 'best_model.pt'), map_location=device)['sens_net']
        sens_net.load_state_dict(state_dict)
    else:
        print(f"Loading checkpoint!")
        sens_net.load_state_dict(ckpt['sens_net'])
        
    engine = Stage2Engine(sens_net=sens_net, model=model)
    
    dset_dict = build_dataset(args, args.num_epochs2)
    train_dl = dset_dict['train_dl']
    val_dl = dset_dict['val_dl']
    augmentor = dset_dict['augmentor']
    
    masker_dict = build_masker(args, args.num_epochs2)
    masker = masker_dict['train_masker']
    val_maskers = masker_dict['val_masker']

    loss_fns = []
    
    for lf in args.losses:
        if lf == "s":
            loss_fn = SSIMLoss().to(device)
        elif lf == 'l':
            loss_fn = nn.L1Loss().to(device)

        loss_fns.append(loss_fn)
    print(f"Loss functions: {loss_fns}")
    
    if args.milestones2:
        milestones = args.milestones2
    else:
        milestones = [i for i in range(0, args.num_epochs, args.step_size)]
    optimizer = optim.Adam(engine.parameters(), args.lr2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones, 
                                               gamma=args.gamma)
    
    print("Training Stage 2")
    print(f"Training {args.model} with {args.sens_model} sensitivity model")
    print(f"Scheduler milestones: {milestones}")
    print(f"Expected last learning rate: {args.lr2 * args.gamma ** len(milestones)}")
    
    train_loop(args=args,
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
               model_path=args.exp_dir2,
               stage=2)

def train_3stage(args, device, ckpt=None):
    model_dict = build_model(args)
    prev_model = model_dict['model2'].to(device)
    model = model_dict['model3'].to(device)
    sens_net = model_dict['sens_net'].to(device)
    
    state_dict = torch.load(os.path.join(args.exp_dir2, 'best_model.pt'), map_location=device)
    sens_net.load_state_dict(state_dict['sens_net'])
    prev_model.load_state_dict(state_dict['model'])
    engine = Stage3Engine(sens_net=sens_net, prev_models=[prev_model], model=model)
        
    dset_dict = build_dataset(args, args.num_epochs3)
    train_dl = dset_dict['train_dl']
    val_dl = dset_dict['val_dl']
    augmentor = dset_dict['augmentor']
    
    masker_dict = build_masker(args, args.num_epochs3)
    masker = masker_dict['train_masker']
    val_maskers = masker_dict['val_masker']

    loss_fns = []
    
    for lf in args.losses:
        if lf == "s":
            loss_fn = SSIMLoss().to(device)
        elif lf == 'l':
            loss_fn = nn.L1Loss().to(device)

        loss_fns.append(loss_fn)
    print(f"Loss functions: {loss_fns}")
    
    if args.milestones3:
        milestones = args.milestones3
    else:
        milestones = [i for i in range(0, args.num_epochs, args.step_size)]
    optimizer = optim.Adam(engine.parameters(), args.lr3)
    
    if ckpt:
        print(f"Loading checkpoint!")
        engine.load_state_dict(ckpt['engine'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones, 
                                               gamma=args.gamma)
    
    print("Training Stage 3")
    print(f"Training {args.model} with {args.sens_model} sensitivity model")
    print(f"Scheduler milestones: {milestones}")
    print(f"Expected last learning rate: {args.lr3 * args.gamma ** len(milestones)}")
    
    train_loop(args=args,
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
               model_path=args.exp_dir3,
               stage=3)
    