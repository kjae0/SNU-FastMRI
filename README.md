
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
전체적인 구조는 아래와 같습니다. <br>
제출 포맷을 최대한 유지하였으나, 알고리즘 특성 상 불가피하게 일부 변경되었습니다.
~~~
/home
└── /Data  # 용량 문제로 실제 데이터는 제출에서 제외했습니다.
    ├── /train
    ├── /val
    └── /leaderboard

/root
    ├── /checkpoints # 제출을 위한 model ckpt와 training loss입니다.
    │   ├── /ckpt1
    │   │   ├── stage2_ckpt.pt
    │   │   ├── stage3_ckpt.pt
    │   │   └── train_loss_log.txt
    │      ....
    ├── /FastMRI_Challenge
    │   ├── /fastmri
    │   ├── /src
    │   ├── /checkpoints
    │   ├── leaderboard_eval.py
    │   ├── leaderboard_eval.sh
    │   ├── reconstruct.py
    │   ├── reconstruct.sh
    │   ├── train.py
    │   ├── train_promptmr_fd_tc.sh
    │   └── train_promptmr_wide_fd_wc.sh
    │      
    ├── /result
    ├── /training logs
    ├── requirements1.txt
    ├── requirements2.txt
    └── README.md
~~~

## Method
2개의 모델의 output을 average ensemble하여 최종 예측값을 내며, 각 모델은 3개의 stage를 통해 개별적으로 학습됩니다. <br>
따라서, 학습되는 모델들은 아래와 같습니다.
~~~
1. model 1 
    1.1. sensitivity network                    < Stage1 >
    1.2. train reconstruction step 1            < Stage2 >
    1.3. train reconstruction step 2            < Stage3 >
    
2. model 2 
    2.1. sensitivity network                    < Stage1 > 
    2.2. train reconstruction step 1            < Stage2 > 
    2.3. train reconstruction step 2            < Stage3 > 
~~~
또한, 최종 2개의 모델들은 다른 resolution을 input으로 하여 학습됩니다. <br>
- kspace의 width를 바탕으로 center crop (crop by width, width crop, wc로 표기됩니다.)
- target의 width를 바탕으로 center crop (target crop, tc로 표기됩니다.)

<strong> 결과적으로, wc, tc에 대한 3 stage, 총 6개의 모델이 학습됩니다. 

최종 선택된 모델들은 아래와 같습니다.</strong>
1. model wide wc (low cascades, wide channels, width crop) - stage3 15 epoch
2. model tc (high cascades, thin channels, target crop) - stage 3 14 epoch


## Get Started
### Train
train.py를 실행하면 되지만, argument가 많은 관계로 ensemble baseline이 되는 2개의 모델의 학습코드를 <br>
- promptmr_fd_tc.sh <br>
- promptmr_wide_fd_wc.sh <br>

으로 저장해두었습니다. 각 실행파일의 구조는 아래와 같습니다. <br>

** 주의사항 **


<strong>
모든 코드들은 실행 전 절대경로를 설정해주시기 바랍니다.<br>
주어진 vessl 서버를 포함한 여러 실험환경에서 반복적으로 실행시켜본 결과,<br>  잘 작동되었으나, 실험환경에 따라 잘 학습되던 중 갑자기 loss가 nan으로 나타날 수도 있습니다.<br> 이 경우, learning rate를 일부 낮추거나, 단순히 이어서 학습할 stage를 contd_stage로 세팅하여 재실행하면 학습이 잘 마무리 되었습니다.<br>
잘 학습이 됨을 증명하기 위해 전체 training log를 첨부하였습니다.<br>

</strong>


~~~
# 예시입니다!
python train.py \
  -b 1 \
  -e1 15 \
  -e2 20 \
  -e3 30 \
  -l1 3e-4 \
  -l2 2e-4 \
  -l3 2e-4 \
  -r 10 \
  -n 'promptmr_fd_tc' \                                                 # result에서 저장될 폴더명입니다.
  -t '/home/Data/train/' \                                              # train data path
  -v '/home/Data/val/' \                                                # validation data path (for train)
  -m 'promptmr' \
  -sm 'promptmr' \
  -s 2 \
  --full_data \
  --crop_by_width \
  --milestones1 '[10, 13, 16, 18]' \
  --milestones2 '[10, 14, 18, 22, 26]' \
  --milestones3 '[10, 13]' \
  --clip 10 \
  --gamma 0.4 \
  --aug_on \
  --mask_type 'equi' \
  --output_target_key image \
  --gpu_id 0 \                                                          # GPU ID
  --aug_delay 3 \
  --sens_chans 3 \
  --sens_num_layers 4 \
  --cascade1 5 \
  --chans1 1 \
  --num_layers1 4 \
  --cascade2 6 \
  --chans2 6 \
  --num_layers2 5 \
  --cascade3 6 \
  --chans3 5 \
  --num_layers3 5 \
  --aug_strength 0.55 \
  --aug_exp_decay 5.0 \
  --aug_weight_translation 0.0 \
  --aug_weight_rotation 0.0 \
  --aug_weight_scaling 1.0 \
  --aug_weight_shearing 0.0 \
  --aug_weight_rot90 0.5 \
  --aug_weight_fliph 0.5 \
  --aug_weight_flipv 0.5 \
  --aug_max_translation_x 0.05 \
  --aug_max_translation_y 0.75 \
  --aug_max_rotation 180 \
  --aug_max_shearing_x 6.25 \
  --aug_max_shearing_y 6.25 \
  --aug_max_scaling 0.25
~~~

training step은 3개의 stage로 이루어져있습니다. 이 3개의 step은 train.py에서 순차적으로 실행됩니다.

1. train sensitivity network   <strong> < Stage1 > </strong> <br>
2. train reconstruction model1  <strong> < Stage2 > </strong> <br>
3. train reconstruction model2   <strong> < Stage3 > </strong> <br>

각 stage의 모델들은 Method에서 언급된 대로 개별적으로 학습되기 때문에 다른 ckpt로 저장됩니다.

### Output Format
~~~
/result
    ├── /{net_name}_stage1
    │   ├── /checkpoints
    │   │   ├── 1_model.pt
    │   │   ├── 2_model.pt
    │   │   ...
    │   │   └── best_model.pt               # 마지막 epoch의 ckpt입니다.
    │   └── train_loss_log.txt 
    │
    ├── /{net_name}_stage2
    │   │   ...
    │   │   ...
    │   └── train_loss_log.txt 
    │
    └── /{net_name}_stage3
        ...
        │   ...
        └── train_loss_log.txt 
~~~


<strong> reconstruction을 위해서는 stage2, stage3의 ckpt를 모두 활용해야 합니다. </strong>


### Reconstruction
제공된 코드와 동일하게 아래 파일을 실행하면 됩니다. <br>
<strong> Vessl 서버 기준 약 2400~2500초가 소요됩니다.</strong>


<strong> 최종 선정되었던 ckpt는 다음과 같습니다. <br>
해당 ckpt들의 경로를 설정해주시면 됩니다. 

- promptmr tc 2stage 20 epoch
- promptmr tc 3stage 14 epoch
- promptmr wide wc 2stage 30 epoch
- promptmr wide wc 3stage 15 epoch <br>
3stage의 경우 overfitting으로 인해 마지막 epoch 모델을 사용하지 않는 다는 점을 주의해주시길 바랍니다.</strong>

~~~
sh reconstruct.sh
~~~

reconstruct.sh에서 실행하는 reconstruct.py의 argument들은 아래와 같습니다.<br>
이것 역시 경로를 잘 설정해주시기 바랍니다.<br>
- -p > data path
- wc_stage2_ckpt_dir > wc model의 stage2 ckpt full path
- wc_stage3_ckpt_dir > wc model의 stage3 ckpt full path
- tc_stage2_ckpt_dir > tc model의 stage2 ckpt full path
- tc_stage2_ckpt_dir > tc model의 stage3 ckpt full path
- output_dir > reconstruction output이 저장될 path <br>

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
제공된 코드와 동일하게 아래 파일을 실행하면 됩니다.
~~~
sh leaderboard_eval.sh
~~~

yp에 reconstruction 시 설정했던 output_dir을 설정해주면 됩니다.
~~~
python leaderboard_eval.py \
  -lp '/home/Data/leaderboard' \
  -yp '/root/reconstructions_tc_wwc'
~~~
