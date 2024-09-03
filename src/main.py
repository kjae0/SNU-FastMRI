import argparse
import yaml
import os
import gc
import torch

from src import train
from src import utils

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_dir', type=str, required=True, help='Path to the config file')
    parser.add_argument('--contd_stage', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default="")
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_arguments()
    cfg = yaml.load(open(args.cfg_dir, 'r'), Loader=yaml.FullLoader)
    
    utils.seed_everything(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("WARNING!\nYou are using CPU for training. This is not recommended.")

    ckpt = None
    if args.ckpt_dir and args.contd_stage != 1:
        print(f"Continuing from {args.ckpt_dir} in stage {args.contd_stage}")
        ckpt = torch.load(os.path.join(args.ckpt_dir, 'checkpoints', 'best_model.pt'))
    elif not args.ckpt_dir and args.contd_stage != 1:
        print(f"Starting from scratch in stage {args.contd_stage}")
    elif args.ckpt_dir and args.contd_stage == 1:
        raise ValueError("ckpt_dir should be given with contd_stage")
    
    if args.contd_stage == 1:
        print("Starting stage 1")
        print("=" * 100)
        cfg['exp_dir'] = os.path.join('../result', str(cfg['net_name'])+"_stage1", 'checkpoints')
        cfg['val_dir'] = os.path.join('../result', str(cfg['net_name'])+"_stage1", 'reconstructions_val')
        cfg['main_dir'] = os.path.join('../result', cfg['net_name'], __file__)
        cfg['val_loss_dir'] = os.path.join('../result', str(cfg['net_name'])+"_stage1")
        
        if os.path.exists(cfg['exp_dir']):
            print(f"WARNING!\nDirectory {cfg['exp_dir']} already exists. I would be OVERWRITING the contents.")
            os.makedirs(cfg['exp_dir'], exist_ok=True)
        
        train.train_1stage(cfg, device, ckpt=ckpt)
        gc.collect()
        torch.cuda.empty_cache()
        args.contd_stage += 1
        
    if args.contd_stage == 2:
        print("Starting stage 2")
        print("=" * 100)
        cfg['exp_dir'] = os.path.join('../result', str(cfg['net_name'])+"_stage2", 'checkpoints')
        cfg['val_dir'] = os.path.join('../result', str(cfg['net_name'])+"_stage2", 'reconstructions_val')
        cfg['main_dir'] = os.path.join('../result', cfg['net_name'], __file__)
        cfg['val_loss_dir'] = os.path.join('../result', str(cfg['net_name'])+"_stage2")
        
        if os.path.exists(cfg['exp_dir']):
            print(f"WARNING!\nDirectory {cfg['exp_dir']} already exists. I would be OVERWRITING the contents.")
            os.makedirs(cfg['exp_dir'], exist_ok=True)
        
        train.train_2stage(cfg, device, ckpt=ckpt)
        gc.collect()
        torch.cuda.empty_cache()
        args.contd_stage += 1
        
    if args.contd_stage == 3:
        print("Starting stage 3")
        print("=" * 100)
        cfg['exp_dir'] = os.path.join('../result', str(cfg['net_name'])+"_stage3", 'checkpoints')
        cfg['val_dir'] = os.path.join('../result', str(cfg['net_name'])+"_stage3", 'reconstructions_val')
        cfg['main_dir'] = os.path.join('../result', cfg['net_name'], __file__)
        cfg['val_loss_dir'] = os.path.join('../result', str(cfg['net_name'])+"_stage3")
        
        if os.path.exists(cfg['exp_dir']):
            print(f"WARNING!\nDirectory {cfg['exp_dir']} already exists. I would be OVERWRITING the contents.")
            os.makedirs(cfg['exp_dir'], exist_ok=True)
        
        train.train_3stage(cfg, device, ckpt=ckpt)
        