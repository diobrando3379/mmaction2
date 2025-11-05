cd /home/diobrando/workspace/mmaction2 && pwd
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python tools/visualizations/vis_cam.py \
  custom/configs/slowfast/slowfast_r50_8xb8-4x16x1-256e_devsign_D_rgb.py \
  work_dirs/slowfast_devsign_r50_4x16x1_256e/best_acc_top1_epoch_205.pth \
  /mnt/e/Dataset/CV_Dataset/DEVISIGN/DEVISIGN_D/P03_1/P03_0006_1_0_20130515.oni/color.avi \
  --device cuda:0 \
  --target-layer-name backbone/fast_path/layer4/2/conv3 \
  --out-filename work_dirs/vis_cam/P03_0006_1_0_20130515.oni_fast_path_layer4_2_conv3.mp4
