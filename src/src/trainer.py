import os
import cv2
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import time
from collections import defaultdict
from torch.utils.data import DataLoader

from src import dataset
from src import utils
from src.loss_function import SSIMLoss, ssim_loss
from src.transforms import DataTransform
from src.augmentor import DataAugmentor
from src.masker import RandomMaskFunc, EquiSpacedMaskFunc
from src.models import promptmr, prompt_sme
from src.models.engine import Stage1Engine, Stage2Engine, Stage3Engine

def build_sensitivity_model(args):
    if not args.crop_by_width:
        opt1 = 1
        opt2 = 2
        opt3 = 1
    else:
        opt1 = 1
        opt2 = 0
        opt3 = 0
    sens_model = prompt_sme.SensitivityModel(num_adj_slices=1,
                                            n_feat0 =int(4*args.sens_chans),
                                            # 6 8 10 origin
                                            # 2 4 6 origin
                                            feature_dim = [int(6*args.sens_chans)+opt1, int(7*args.sens_chans)+opt2, int(9*args.sens_chans)+opt3],
                                            prompt_dim = [int(2*args.sens_chans)+opt1, int(3*args.sens_chans)+opt2, int(5*args.sens_chans)+opt3],
                                            len_prompt = [5, 5, 5],
                                            prompt_size = [8, 4, 2],
                                            n_enc_cab = [2, 3, 3],
                                            n_dec_cab = [2, 2, 3],
                                            n_skip_cab = [1, 1, 1],
                                            n_bottleneck_cab = 3,
                                            no_use_ca = None,
                                            mask_center = True,
                                            low_mem = False
                                            )
    return sens_model

def build_recon_model(num_cascades, chans):
    model = promptmr.PromptMR(
        num_cascades = num_cascades,
        num_adj_slices = 1,
        n_feat0 = 8*chans,
        feature_dim = [12*chans, 16*chans, 20*chans],
        prompt_dim = [4*chans, 8*chans, 12*chans],
        len_prompt = [5, 5, 5],
        prompt_size = [8*chans, 4*chans, 2*chans],
        n_enc_cab = [2, 3, 3],
        n_dec_cab = [2, 2, 3],
        n_skip_cab = [1, 1, 1],
        n_bottleneck_cab = 3
    )
    
    return model
    
def build_model(args):
    sens_model = build_sensitivity_model(args)
    model1 = build_recon_model(args.cascade1, args.chans1)
    model2 = build_recon_model(args.cascade2, args.chans2)
    model3 = build_recon_model(args.cascade3, args.chans3)

    return {'sens_net': sens_model,
            'model1': model1,
            'model2': model2,
            'model3': model3}

def build_dataset(args, num_epochs):
    augmentor = DataAugmentor(args, num_epochs)
    train_transform = DataTransform(False, args.max_key, augmentor, crop_by_width=args.crop_by_width)

    train_dataset = dataset.SliceData(root=[args.data_path_train, args.data_path_val], 
                                    transform=train_transform,
                                    input_key=args.input_key, 
                                    target_key=args.target_key)

    train_dl = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers)
    
    return {'augmentor': augmentor, 
            'train_dl': train_dl, 
            'val_dl': None}

def build_masker(args, num_epochs):
    masker = None
    val_maskers = {}

    if not args.mask_off:
        print(f"Masking with {args.mask_type} mask")
        print(f"Center fraction: {args.center_fraction}, Acceleration: {args.acceleration}")
        
        if args.mask_type.lower() == 'random':
            masker = RandomMaskFunc(args.center_fraction, args.acceleration)
        elif args.mask_type.lower() == 'equi':
            if args.acc_scheduler:
                masker = EquiSpacedMaskFunc(args.center_fraction, args.acceleration, start_ratio=args.acc_scheduler_t0, n_steps=int(num_epochs * args.acc_scheduler_tmax))
            else:
                masker = EquiSpacedMaskFunc(args.center_fraction, args.acceleration)
        else:
            raise NotImplementedError("Other mask is not implemented yet.")
        
    print("Validation Mask Setting")
    print(f"Center fraction: {args.val_center_fraction}, Acceleration: {args.val_acceleration}")
    
    for i in range(len(args.val_center_fraction)):
        val_maskers[args.val_acceleration[i]] = EquiSpacedMaskFunc([args.val_center_fraction[i]], [args.val_acceleration[i]])
        
    return {'train_masker': masker, 
            'val_masker': val_maskers}

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
        
        if loss > 0.3:
            print(loss)
        
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

        # if iter == 20:
        #     break
            
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
    
    # if ckpt:
    #     print(f"Loading checkpoint!")
    #     engine.load_state_dict(ckpt['engine'])
    #     optimizer.load_state_dict(ckpt['optimizer'])
    
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
    