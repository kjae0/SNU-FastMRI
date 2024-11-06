import argparse
import yaml
import os
import gc
import torch

from src import train
from src import trainer
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
        print("[INFO] WARNING!\nYou are using CPU for training. This is not recommended.")

    ckpt = None
    if args.ckpt_dir and args.contd_stage != 1:
        print(f"[INFO] Continuing from {args.ckpt_dir} in stage {args.contd_stage}")
        ckpt = torch.load(os.path.join(args.ckpt_dir, 'checkpoints', 'best_model.pt'))
    elif not args.ckpt_dir and args.contd_stage != 1:
        print(f"[INFO] Starting from scratch in stage {args.contd_stage}")
    elif args.ckpt_dir and args.contd_stage == 1:
        raise ValueError("ckpt_dir should be given with contd_stage")
    
    train_func_dict = {
        1: train.train_1stage,
        2: train.train_2stage,
        3: train.train_3stage
    }
    
    for i in range(args.contd_stage, 4):
        runner = trainer.Trainer(cfg, device)
        
        print(f"[INFO] Starting stage {args.contd_stage}")
        print("=" * 100)
        cfg[f'stage{args.contd_stage-1}_exp_dir'] = os.path.join('../result', str(cfg['net_name'])+f"_stage{args.contd_stage-1}", 'checkpoints')
        cfg['exp_dir'] = os.path.join('../result', str(cfg['net_name'])+f"_stage{args.contd_stage}", 'checkpoints')
        cfg['val_dir'] = os.path.join('../result', str(cfg['net_name'])+f"_stage{args.contd_stage}", 'reconstructions_val')
        cfg['main_dir'] = os.path.join('../result', cfg['net_name'], __file__)
        cfg['val_loss_dir'] = os.path.join('../result', str(cfg['net_name'])+f"_stage{args.contd_stage}")
        
        if os.path.exists(cfg['exp_dir']):
                print(f"WARNING!\nDirectory {cfg['exp_dir']} already exists. I would be OVERWRITING the contents.")
                os.makedirs(cfg['exp_dir'], exist_ok=True)
                
        # train_func_dict[i](cfg, device, ckpt=ckpt)
        runner.run_training(i, ckpt=ckpt)
        gc.collect()
        torch.cuda.empty_cache()
        args.contd_stage += 1
        