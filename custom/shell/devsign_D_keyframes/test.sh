cd /home/diobrando/workspace/mmaction2 && pwd
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
CKPT="/home/diobrando/workspace/mmaction2/work_dirs/slowfast_CSL_SLR250_r50_4x16x1_256e/best_acc_top1_epoch_112.pth"

python tools/test.py \
  custom/configs/slowfast/slowfast_r50_8xb8-4x16x1-256e_CSL_SLR250.py \
  "$CKPT" \
  --work-dir work_dirs/slowfast_test_CSL_SLR250_r50_4x16x1_256e

# 11/06 09:38:35 - mmengine - INFO - Epoch(test) [2500/2500]    test/acc/top1: 0.8024  test/acc/top5: 0.9644  test/acc/mean1: 0.8024  test/data_time: 0.0623  test/time: 0.3158