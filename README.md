
# 2024 SNU FastMRI Challenge
Submission Code for Team '재애애영' <br>Members: 김재영, 구자혁 <br>
Final SSIM (leaderboard) - 0.9824 <br>

## Environments
~~~
GPU : NVIDIA GeForce RTX 2080 Ti 
CPU : Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
package version : requirements.txt
python 3.8.19
pytorch 2.4.0+cu121
CUDA 12.3
~~~

## Overview
Our final submission is average ensemble of two models, which are two PromptMR models trained by different data resolutions. (input width, target width) <br>

~~~
  1. sensitivity network                    < Stage1 >
  2. train reconstruction step 1            < Stage2 >
  3. train reconstruction step 2            < Stage3 >
~~~
Each models are trained sequentially due to GPU 8GB VRAM limitation.
Stage1 -> train sensitivity network
Stage2 -> train first reconstruction network with sensitivity network from stage1 frozen.
Stage3 -> train second reconstruction network with sensitivity network from stage1 and reconstruction network from stage frozen. <br>
second reconstruction network gets output of first reconstruction network as a input.

## Get Started
### Train

- promptmr target width crop (tc)
~~~
python train.py \
~~~

- promptmr input width crop (wc)
~~~
python train.py \
~~~

Training step consists of 3 stages, each steps are executed sequentially in 'train.py'.

1. train sensitivity network   <strong> < Stage1 > </strong> <br>
2. train reconstruction model1  <strong> < Stage2 > </strong> <br>
3. train reconstruction model2   <strong> < Stage3 > </strong> <br>
Checkpoints of each stages are saved seperately.

### Reconstruction
<strong> The best checkpoint of each models are following, <br>
- promptmr tc 2stage 20 epoch
- promptmr tc 3stage 14 epoch
- promptmr wide wc 2stage 30 epoch
- promptmr wide wc 3stage 15 epoch <br>
</strong>

~~~
sh reconstruct.sh
~~~

Arguments for reconstruction are following, <br>
- -p > data path
- wc_stage2_ckpt_dir > wc model의 stage2 ckpt full path
- wc_stage3_ckpt_dir > wc model의 stage3 ckpt full path
- tc_stage2_ckpt_dir > tc model의 stage2 ckpt full path
- tc_stage2_ckpt_dir > tc model의 stage3 ckpt full path
- output_dir > path for reconstruction output <br>

~~~ 
python reconstruct.py \
  -b 2 \
  -p '/home/diya/Public/Image2Smiles/jy/fastmri/data/Data/leaderboard' \
  -g 1 \
  --wc_stage2_ckpt_dir ../checkpoints/promptmr_wide_fd_wc/best_model_stage2.pt \
  --wc_stage3_ckpt_dir ../checkpoints/promptmr_wide_fd_wc/best_model_stage3.pt \
  --tc_stage2_ckpt_dir ../checkpoints/promptmr_fd_tc/best_model_stage2.pt \
  --tc_stage3_ckpt_dir ../checkpoints/promptmr_fd_tc/best_model_stage3.pt \
  --output_dir ../reconstructions_tc_wwc \
  --wc_cascade2 6 \
  --wc_chans2 6 \
  --wc_cascade3 6\
  --wc_chans3 5 \
  --tc_cascade2 8 \
  --tc_chans2 4 \
  --tc_cascade3 8\
  --tc_chans3 4 \
  --sens_chans 3 \
  --stage 3 
~~~

### Evaluation
~~~
sh leaderboard_eval.sh
~~~

~~~
python leaderboard_eval.py \
  -lp '/home/Data/leaderboard' \
  -yp '/root/reconstructions_tc_wwc'
~~~

### Attribution
Parts of this repository are based on or include code from the following sources:

- [PromptMR](https://github.com/hellopipu/PromptMR)
  (paper: Fill the K-Space and Refine the Image: Prompting for Dynamic and Multi-Contrast MRI Reconstruction <br>
  https://arxiv.org/abs/2309.13839)  

- [FastMRI](https://github.com/facebookresearch/fastMRI)  

- [FastMRI SNU](https://github.com/LISTatSNU/FastMRI_challenge)  

