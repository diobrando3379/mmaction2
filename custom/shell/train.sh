cd /home/diobrando/workspace/mmaction2 && pwd
python tools/train.py \
  custom/configs/slowfast/slowfast_r50_8xb8-4x16x1-256e_devsign_D_rgb.py \
  --work-dir work_dirs/slowfast_devsign_r50_4x16x1_256e
#   --seed 0 --deterministic