_base_ = [
    './uvtr_c_r50_h11.py'
]

model = dict(
    img_backbone=dict(depth=101)
    )

load_from='ckpts/uvtr/pretrain/fcos3d.pth' # please download the pretrained model from the our git