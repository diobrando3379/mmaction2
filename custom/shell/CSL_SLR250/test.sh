cd /home/diobrando/workspace/mmaction2 && pwd
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
CKPT="/home/diobrando/workspace/mmaction2/work_dirs/slowfast_devsign_r50_4x16x1_256e/best_acc_top1_epoch_205.pth"

python tools/test.py \
  custom/configs/slowfast/slowfast_r50_8xb8-4x16x1-256e_devsign_D_rgb.py \
  "$CKPT" \
  --work-dir work_dirs/slowfast_devsign_r50_4x16x1_256e