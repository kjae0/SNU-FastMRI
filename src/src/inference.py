from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from src.utils import save_reconstructions
from src.dataset_leaderboard import create_data_loaders
from src.models import engine, promptmr

import os
import warnings
import torch
import numpy as np

warnings.filterwarnings("ignore")

def build_inference_model(sens_chans, cascade2, chans2, cascade3, chans3, crop_by_width, stage):
    if crop_by_width:
        opt1 = 1
        opt2 = 0
        opt3 = 0
    else:
        opt1 = 1
        opt2 = 2
        opt3 = 1
    
    sens_net = promptmr.SensitivityModel(num_adj_slices = 1,
                                                n_feat0 = 4*sens_chans,
                                                feature_dim = [6*sens_chans+opt1, 7*sens_chans+opt2, 9*sens_chans+opt3],
                                                prompt_dim = [2*sens_chans+opt1, 3*sens_chans+opt2, 5*sens_chans+opt3],
                                                len_prompt = [5, 5, 5],
                                                prompt_size = [8, 4, 2],
                                                n_enc_cab = [2, 3, 3],
                                                n_dec_cab = [2, 2, 3],
                                                n_skip_cab = [1, 1, 1],
                                                n_bottleneck_cab = 3,
                                                no_use_ca = None,
                                                mask_center = True,
                                                low_mem = False)
    
    if stage == 2:    
        current_model = promptmr.PromptMR(num_cascades = cascade2,
                                            num_adj_slices = 1,
                                            n_feat0 = 8*chans2,
                                            feature_dim = [12*chans2, 16*chans2, 20*chans2],
                                            prompt_dim = [4*chans2, 8*chans2, 12*chans2],
                                            len_prompt = [5, 5, 5],
                                            prompt_size = [8*chans2, 4*chans2, 2*chans2],
                                            n_enc_cab = [2, 3, 3],
                                            n_dec_cab = [2, 2, 3],
                                            n_skip_cab = [1, 1, 1],
                                            n_bottleneck_cab = 3)
        
        model = engine.Stage2Engine(sens_net, current_model)
    elif stage == 3:    
        prev_model = promptmr.PromptMR(num_cascades = cascade2,
                                        num_adj_slices = 1,
                                        n_feat0 = 8*chans2,
                                        feature_dim = [12*chans2, 16*chans2, 20*chans2],
                                        prompt_dim = [4*chans2, 8*chans2, 12*chans2],
                                        len_prompt = [5, 5, 5],
                                        prompt_size = [8*chans2, 4*chans2, 2*chans2],
                                        n_enc_cab = [2, 3, 3],
                                        n_dec_cab = [2, 2, 3],
                                        n_skip_cab = [1, 1, 1],
                                        n_bottleneck_cab = 3)
        
        current_model = promptmr.PromptMR(num_cascades = cascade3,
                                            num_adj_slices = 1,
                                            n_feat0 = 8*chans3,
                                            feature_dim = [12*chans3, 16*chans3, 20*chans3],
                                            prompt_dim = [4*chans3, 8*chans3, 12*chans3],
                                            len_prompt = [5, 5, 5],
                                            prompt_size = [8*chans3, 4*chans3, 2*chans3],
                                            n_enc_cab = [2, 3, 3],
                                            n_dec_cab = [2, 2, 3],
                                            n_skip_cab = [1, 1, 1],
                                            n_bottleneck_cab = 3)
        
        model = engine.Stage3Engine(sens_net, [prev_model], current_model)
    
    return model
    
def to_device(model, device, stage):
    model.sens_net.to(device)
    model.model.to(device)
    
    if stage > 2:
        print("stage > 2")
        for prev_model in model.prev_models:
            prev_model.to(device)
        model = model.to(device)
    
    return model
    
def test(args, wc_model, tc_model, data_loader):
        
    reconstructions = defaultdict(dict)

    idx = 0
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in tqdm(data_loader, total=len(data_loader), ncols=80):
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)    
            
            output = (wc_model(kspace, mask, 0)['image'] + tc_model(kspace, mask, 0)['image']) / 2
            
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
            
            # if idx == 10:
            #     break
            # idx += 1
            
    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )

    return reconstructions, None, None

def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())
                
    print(f"Loading model")
    wc_model = build_inference_model(sens_chans=args.sens_chans,
                                     cascade2=args.wc_cascade2,
                                     chans2=args.wc_chans2,
                                     cascade3=args.wc_cascade3,
                                     chans3=args.wc_chans3,
                                     crop_by_width=True,
                                     stage=args.wc_stage)
    tc_model = build_inference_model(sens_chans=args.sens_chans,
                                     cascade2=args.tc_cascade2,
                                     chans2=args.tc_chans2,
                                     cascade3=args.tc_cascade3,
                                     chans3=args.tc_chans3,
                                     crop_by_width=False,
                                     stage=args.tc_stage)
    
    wc_stage2_ckpt = torch.load(args.wc_stage2_ckpt_dir, map_location=device)
    wc_stage3_ckpt = torch.load(args.wc_stage3_ckpt_dir, map_location=device)
    
    tc_stage2_ckpt = torch.load(args.tc_stage2_ckpt_dir, map_location=device)
    tc_stage3_ckpt = torch.load(args.tc_stage3_ckpt_dir, map_location=device)
    
    print(wc_stage2_ckpt['epoch'])
    print(wc_stage3_ckpt['epoch'])
    print(tc_stage2_ckpt['epoch'])
    print(tc_stage3_ckpt['epoch'])

    if args.wc_stage == 3:
        print("wc sens net loaded")
        print("wc model loaded")
        print("wc stage 3")
        
        wc_model.sens_net.load_state_dict(wc_stage3_ckpt['sens_net'])
        wc_model.model.load_state_dict(wc_stage3_ckpt['model'])
        
        for prev_model in wc_model.prev_models:
            prev_model.load_state_dict(wc_stage2_ckpt['model'])
        print("wc prev model loaded")
    elif args.wc_stage == 2:
        print("wc sens net loaded")
        print("wc model loaded")
        print("wc stage 2")
        
        wc_model.sens_net.load_state_dict(wc_stage2_ckpt['sens_net'])
        wc_model.model.load_state_dict(wc_stage2_ckpt['model'])
        print("wc model loaded")
    
    if args.tc_stage == 3:
        print("tc sens net loaded")
        print("tc model loaded")
        print("tc stage 3")
        
        tc_model.sens_net.load_state_dict(tc_stage3_ckpt['sens_net'])
        tc_model.model.load_state_dict(tc_stage3_ckpt['model'])
        
        for prev_model in tc_model.prev_models:
            prev_model.load_state_dict(tc_stage2_ckpt['model'])
        print("tc prev model loaded")
    elif args.tc_stage == 2:
        print("tc sens net loaded")
        print("tc model loaded")
        print("tc stage 2")
        
        tc_model.sens_net.load_state_dict(tc_stage2_ckpt['sens_net'])
        tc_model.model.load_state_dict(tc_stage2_ckpt['model'])
        print("tc model loaded")

    wc_model = to_device(wc_model, device, args.wc_stage)
    tc_model = to_device(tc_model, device, args.tc_stage)
    
    print('Model loaded successfully')
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs, kspace_predictions = test(args, wc_model, tc_model, forward_loader)
    save_reconstructions(reconstructions, Path(args.save_dir), inputs=inputs, kspace=kspace_predictions)
    