import argparse
import shutil
import os, sys, ast
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
import numpy as np

from src import dataset
from src import trainer
from src import utils
from src.transforms import DataTransform
from src.augmentor import DataAugmentor

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    
    parser.add_argument('-e1', '--num-epochs1', type=int, default=5, help='Number of epochs')
    parser.add_argument('-e2', '--num-epochs2', type=int, default=5, help='Number of epochs')
    parser.add_argument('-e3', '--num-epochs3', type=int, default=5, help='Number of epochs')
    
    parser.add_argument('-l1', '--lr1', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('-l2', '--lr2', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('-l3', '--lr3', type=float, default=5e-4, help='Learning rate')
    
    parser.add_argument('-r', '--report-interval', type=int, default=10, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-sm', '--sens_model', type=str, required=True)
    parser.add_argument('-g', '--gamma', type=float, default=0.5)
    parser.add_argument('-s', '--step_size', type=int, required=True)
    parser.add_argument('--clip_norm', type=int, default=10)
    parser.add_argument('--milestones1', type=arg_as_list, default=[])
    parser.add_argument('--milestones2', type=arg_as_list, default=[])
    parser.add_argument('--milestones3', type=arg_as_list, default=[])

    parser.add_argument('--ckpt_dir', type=str, default="")
    parser.add_argument('--contd_stage', type=int, default=1)
    
    parser.add_argument('--losses', type=str, default="s")
    parser.add_argument('--output_target_key', type=str, default="image")
    
    parser.add_argument('--cascade1', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans1', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--num_layers1', type=int, default=4) ## important hyperparameter
    
    parser.add_argument('--cascade2', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans2', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--num_layers2', type=int, default=4) ## important hyperparameter
    
    parser.add_argument('--cascade3', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans3', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--num_layers3', type=int, default=4) ## important hyperparameter
    
    parser.add_argument('--sens_chans', type=float, default=6, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter 
    parser.add_argument('--sens_num_layers', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    
    parser.add_argument('--input_key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target_key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max_key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--num_workers', type=int, default=1, help='Set num workers')
    parser.add_argument('--crop_by_width', default=False, action='store_true', help='Set num workers')
    parser.add_argument('--full_data', default=False, action='store_true', help='Set num workers')
    parser.add_argument('--gpu_id', type=int, default=0, help='set gpu id')

    # augmentation setting
    parser.add_argument('--aug_on', default=False, help='This switch turns data augmentation on.', action='store_true')
    parser.add_argument('--aug_schedule', type=str, default='exp', help='Type of data augmentation strength scheduling. Options: constant, ramp, exp')
    parser.add_argument('--aug_delay', type=int, default=2, help='Number of epochs at the beginning of training without data augmentation. The schedule in --aug_schedule will be adjusted so that at the last epoch the augmentation strength is --aug_strength.')
    parser.add_argument('--aug_strength', type=float, default=0.6, help='Augmentation strength, combined with --aug_schedule determines the augmentation strength in each epoch')
    parser.add_argument('--aug_exp_decay', type=float, default=0.1, help='Exponential decay coefficient if --aug_schedule is set to exp. 1.0 is close to linear, 10.0 is close to step function')

    parser.add_argument('--aug_weight_translation', type=float, default=1.0, help='Weight of translation probability. Augmentation probability will be multiplied by this constant')
    parser.add_argument('--aug_weight_rotation', type=float, default=1.0, help='Weight of rotation probability. Augmentation probability will be multiplied by this constant')
    parser.add_argument('--aug_weight_scaling', type=float, default=1.0, help='Weight of scaling probability. Augmentation probability will be multiplied by this constant')
    parser.add_argument('--aug_weight_shearing', type=float, default=1.0, help='Weight of shearing probability. Augmentation probability will be multiplied by this constant')
    parser.add_argument('--aug_weight_rot90', type=float, default=1.0, help='Weight of rot90 probability. Augmentation probability will be multiplied by this constant')
    parser.add_argument('--aug_weight_fliph', type=float, default=1.0, help='Weight of fliph probability. Augmentation probability will be multiplied by this constant')
    parser.add_argument('--aug_weight_flipv', type=float, default=1.0, help='Weight of flipv probability. Augmentation probability will be multiplied by this constant')

    parser.add_argument('--aug_upsample', default=False, action='store_true', help='Set to upsample before augmentation to avoid aliasing artifacts. Adds heavy extra computation.')
    parser.add_argument('--aug_upsample_factor', type=int, default=2, help='Factor of upsampling before augmentation, if --aug_upsample is set')
    parser.add_argument('--aug_upsample_order', type=int, default=1, help='Order of upsampling filter before augmentation, 1: bilinear, 3:bicubic')
    parser.add_argument('--aug_interpolation_order', type=int, default=1, help='Order of interpolation filter used in data augmentation, 1: bilinear, 3:bicubic. Bicubic is not supported yet.')

    parser.add_argument('--aug_max_translation_x', type=float, default=0.125, help='Maximum translation applied along the x axis as fraction of image width')
    parser.add_argument('--aug_max_translation_y', type=float, default=0.125, help='Maximum translation applied along the y axis as fraction of image height')
    parser.add_argument('--aug_max_rotation', type=float, default=180., help='Maximum rotation applied in either clockwise or counter-clockwise direction in degrees.')
    parser.add_argument('--aug_max_shearing_x', type=float, default=15.0, help='Maximum shearing applied in either positive or negative direction in degrees along x axis.')
    parser.add_argument('--aug_max_shearing_y', type=float, default=15.0, help='Maximum shearing applied in either positive or negative direction in degrees along y axis.')
    parser.add_argument('--aug_max_scaling', type=float, default=0.25, help='Maximum scaling applied as fraction of image dimensions. If set to s, a scaling factor between 1.0-s and 1.0+s will be applied.')    
    parser.add_argument("--max_train_resolution", nargs="+", default=None, type=int, help="If given, training slices will be center cropped to this size if larger along any dimension.",)

    # masking setting
    parser.add_argument('--mask_off', default=False, action="store_true", help='masking on or off')
    parser.add_argument('--acceleration_scaler', default=False, action="store_true", help='masking on or off')
    parser.add_argument('--mask_type', type=str, default='equi', help='select masking type')
    parser.add_argument('--center_fraction', type=list, default=[0.08 for _ in range(13)], help='center_fraction list')
    parser.add_argument('--acceleration', type=list, default=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], help='acceleration list')
    parser.add_argument('--acc_scheduler', default=True, action='store_false', help='acceleration list')
    parser.add_argument('--acc_scheduler_t0', type=float, default=0.4, help='acceleration list')
    parser.add_argument('--acc_scheduler_tmax', type=float, default=0.5, help='전체 에폭에서의 비율임. 0.5면 50%에서 최대값을 가짐 이거 따라 lr scheduler도 조정해야함')
    
    parser.add_argument('--val_center_fraction', type=list, default=[0.08, 0.08, 0.08, 0.08, 0.08, 0.08], help='center_fraction list')
    parser.add_argument('--val_acceleration', type=list, default=[4, 5, 8, 9, 12 ,16], help='acceleration list')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    utils.seed_everything(args.seed)

    # GPU setting
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    args.exp_dir1 = os.path.join('../result', str(args.net_name)+"_stage1", 'checkpoints')
    args.exp_dir2 = os.path.join('../result', str(args.net_name)+"_stage2", 'checkpoints')
    args.exp_dir3 = os.path.join('../result', str(args.net_name)+"_stage3", 'checkpoints')
    
    args.val_dir1 = os.path.join('../result', str(args.net_name)+"_stage1", 'reconstructions_val')
    args.val_dir2 = os.path.join('../result', str(args.net_name)+"_stage2", 'reconstructions_val')
    args.val_dir3 = os.path.join('../result', str(args.net_name)+"_stage3", 'reconstructions_val')
    
    args.main_dir = '../result' / args.net_name / __file__
    
    args.val_loss_dir1 = os.path.join('../result', str(args.net_name)+"_stage1")
    args.val_loss_dir2 = os.path.join('../result', str(args.net_name)+"_stage2")
    args.val_loss_dir3 = os.path.join('../result', str(args.net_name)+"_stage3")

    if os.path.exists(args.exp_dir1):
        print(f"WARNING!\nDirectory {args.exp_dir1} already exists. I would be OVERWRITING the contents.")
        
    if os.path.exists(args.val_dir1):
        print(f"WARNING!\nDirectory {args.val_dir1} already exists. I would be OVERWRITING the contents.")
        
    if args.contd_stage == 1:
        print("Creating directories")
        os.makedirs(args.exp_dir1, exist_ok=True)
        os.makedirs(args.exp_dir2, exist_ok=True)
        os.makedirs(args.exp_dir3, exist_ok=True)
        # args.val_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("Continuing from previous stage")

    ckpt = None
    if args.ckpt_dir and args.contd_stage != 1:
        print(f"Continuing from {args.ckpt_dir} in stage {args.contd_stage}")
        ckpt = torch.load(os.path.join(args.ckpt_dir, 'checkpoints', 'best_model.pt'))
    elif not args.ckpt_dir and args.contd_stage != 1:
        print(f"Starting from scratch in stage {args.contd_stage}")
    elif args.ckpt_dir and args.contd_stage == 1:
        raise ValueError("ckpt_dir and contd_stage should be given together")

    import gc
    
    if args.contd_stage == 1:
        trainer.train_1stage(args, device, ckpt=ckpt)
        gc.collect()
        torch.cuda.empty_cache()
        args.contd_stage += 1
        
    if args.contd_stage == 2:
        trainer.train_2stage(args, device, ckpt=ckpt)
        gc.collect()
        torch.cuda.empty_cache()
        args.contd_stage += 1
        
    if args.contd_stage == 3:
        trainer.train_3stage(args, device, ckpt=ckpt)
        