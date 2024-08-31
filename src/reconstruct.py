from pathlib import Path

import argparse
import os
import sys
import time
    
from src import inference

def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
        
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')
    
    parser.add_argument("--wc_stage2_ckpt_dir", type=str, required=True)
    parser.add_argument("--wc_stage3_ckpt_dir", type=str, required=True)
  
    parser.add_argument("--tc_stage2_ckpt_dir", type=str, required=True)
    parser.add_argument("--tc_stage3_ckpt_dir", type=str, required=True)
    
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument('--wc_cascade2', type=int, required=True)
    parser.add_argument('--wc_chans2', type=int, required=True)
  
    parser.add_argument('--wc_cascade3', type=int, required=True)
    parser.add_argument('--wc_chans3', type=int, required=True)
    
    parser.add_argument('--tc_cascade2', type=int, required=True)
    parser.add_argument('--tc_chans2', type=int, required=True)
  
    parser.add_argument('--tc_cascade3', type=int, required=True)
    parser.add_argument('--tc_chans3', type=int, required=True)
    
    parser.add_argument('--sens_chans', type=int, default=3, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter 
    parser.add_argument('--tc_stage', type=int, default=3, help='Number of cascades | Should be less than 12')
    parser.add_argument('--wc_stage', type=int, default=3, help='Number of cascades | Should be less than 12')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    
    public_acc, private_acc = None, None

    assert(len(os.listdir(args.path_data)) == 2)

    for acc in os.listdir(args.path_data):
      if acc in ['acc4', 'acc5', 'acc8']:
        public_acc = acc
      else:
        private_acc = acc
        
    assert(None not in [public_acc, private_acc])
    
    start_time = time.time()
    
    # Public Acceleration
    args.data_path = args.path_data / public_acc # / "kspace"    
    args.save_dir = os.path.join(args.output_dir, 'public')
    print(f'Saved into {args.save_dir}')
    inference.forward(args)
    
    # Private Acceleration
    args.data_path = args.path_data / private_acc # / "kspace"    
    args.save_dir = os.path.join(args.output_dir, 'private')
    print(f'Saved into {args.save_dir}')
    inference.forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')
    
    print('Success!') if reconstructions_time < 3000 else print('Fail!')
    