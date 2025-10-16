# 在ubuntu 5070Ti上安装

```bash
conda create -n dformer python=3.10 -y
conda activate dformer
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -U openmim
mim install mmcv==2.1.0
mim install mmengine
cd mmaction2
pip install -v -e .
pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex
pip install "importlib-metadata>=6,<7"
```

```bash
# 验证环境安装正确性
python -c "import sys,torch,mmcv,mmaction,decord; print('python',sys.version); print('torch',torch.__version__); print('mmcv',mmcv.__version__); print('mmaction',getattr(mmaction,'__version__','?')); print('decord',decord.__version__)"

python -c "import importlib.util, mmaction, os; print('mmaction path:', os.path.dirname(mmaction.__file__)); print('editable:', importlib.util.find_spec('mmaction').origin)"

mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .

# demo.mp4 和 label_map_k400.txt 都来自于 Kinetics-400
python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
```
