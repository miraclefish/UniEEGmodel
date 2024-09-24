model_name=UniEEG
#exp_name=UniTS_supervised_x64
wandb_mode=online
project_name=TUAR_sup

d_model=512
random_port=$((RANDOM % 9000 + 1000))

export WANDB_API_KEY="ad9d2816977a1bcf87acf8a2763df7bc8fdc155a"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

# Supervised learning
/home/ps/anaconda3/envs/time_series/bin/torchrun --nnodes 1 --nproc-per-node=4  --master_port $random_port  run.py \
  --ddp \
  --try_run \
  --is_training 1 \
  --model $model_name \
  --lradj supervised \
  --prompt_num 8 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 8 \
  --d_model $d_model \
  --des 'Exp' \
  --acc_it 1 \
  --learning_rate 3e-4 \
  --weight_decay 0 \
  --prompt_tune_epoch 10 \
  --train_epochs 10 \
  --batch_size 256 \
  --debug $wandb_mode \
  --project_name $project_name \
  --task_data_config_path  data_provider/multi_task.yaml \
  --pretrained_weight 'checkpoints/ALL_UniEEG_all_dm512_el8_Exp_07_26_11_53_58/pretrain_checkpoint.pth' > current.log 2>&1 &