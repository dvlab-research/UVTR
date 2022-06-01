from re import I
from collections import OrderedDict
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
from mmdet3d.core import bbox3d2result
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.core.merge_all_augs import merge_all_aug_bboxes_3d

from ..utils import Uni3DViewTrans

@DETECTORS.register_module()
class UVTRKDM(MVXTwoStageDetector):
    """UVTRKDM."""
    def __init__(self,
                 distill_type='Multi_2D',
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 depth_head=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 teacher_view_head=None,
                 teacher_pts_middle_encoder=None,
                 teacher_pts_backbone=None,
                 teacher_pts_neck=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 pretrained_img=None,
                 pretrained_teacher=None,
                 load_img=None,
                 load_teacher=None):
        super(UVTRKDM,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        if self.with_img_backbone:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
            in_channels = self.img_neck.out_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels  = in_channels[0]
            self.input_proj = Conv2d(in_channels, out_channels, kernel_size=1)
            if "SimpleDepth" in depth_head.type:
                self.depth_dim = depth_head.model.depth_dim
                self.depth_net = Conv2d(out_channels, self.depth_dim, kernel_size=1)
            else:
                raise NotImplementedError
            self.depth_head = depth_head
            self.use_grid_mask = use_grid_mask
        
        if '_2D' not in distill_type:
            self.img_backbone = None
            self.img_neck = None
            self.use_grid_mask = False

        # build encoder and head for techer
        if img_backbone:
            self.teacher_img_backbone = builder.build_backbone(img_backbone)
        if img_neck:
            self.teacher_img_neck = builder.build_neck(img_neck)
        if teacher_view_head.get("view_cfg", None):
            self.teacher_trans = Uni3DViewTrans(**teacher_view_head.view_cfg)
        if teacher_view_head.get("unified_conv", None):
            self.unified_conv = []
            for k in range(teacher_view_head.unified_conv['num_conv']):
                conv = nn.Sequential(
                            nn.Conv3d(teacher_view_head.view_cfg['embed_dims'],
                                      teacher_view_head.view_cfg['embed_dims'],
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=True),
                            nn.BatchNorm3d(teacher_view_head.view_cfg['embed_dims']),
                            nn.ReLU(inplace=True))
                self.add_module("{}_head_{}".format('conv_trans', k + 1), conv)
                self.unified_conv.append(conv)
        
        self.teacher_input_proj = Conv2d(in_channels, out_channels, kernel_size=1)
        self.teacher_depth = Conv2d(out_channels, self.depth_dim, kernel_size=1)

        if teacher_pts_middle_encoder:
            self.teacher_pts_middle_encoder = builder.build_middle_encoder(
                teacher_pts_middle_encoder)
        if teacher_pts_backbone:
            self.teacher_pts_backbone = builder.build_backbone(teacher_pts_backbone)
        if teacher_pts_neck is not None:
            self.teacher_pts_neck = builder.build_neck(teacher_pts_neck)

        self.distill_type = distill_type
        self.pretrained_img = pretrained_img
        self.pretrained_teacher = pretrained_teacher
        self.load_img = load_img
        self.load_teacher = load_teacher

    def init_weights(self):
        """Initialize weights of the depth head."""
        # load pretrained image model
        if self.pretrained_img is not None:
            ckpt_load = torch.load(self.pretrained_img, 
                           map_location="cuda:{}".format(torch.cuda.current_device()))["state_dict"]
            print("Loaded pretrained model from: {}".format(self.pretrained_img))
            for img_key in self.load_img:
                dict_load = {_key.replace(img_key+'.',''):ckpt_load[_key] 
                            for _key in ckpt_load if img_key in _key}
                getattr(self, img_key).load_state_dict(dict_load, strict=False)
                print("Loaded pretrained {}".format(img_key))
                assert len(dict_load) > 0

        # load pretrained sweep model
        if self.pretrained_teacher is not None:
            ckpt_load = torch.load(self.pretrained_teacher, 
                           map_location="cuda:{}".format(torch.cuda.current_device()))["state_dict"]
            print("Loaded pretrained teacher model from: {}".format(self.pretrained_teacher))

            # load pts key for teacher
            for load_key in self.load_teacher:
                teacher_key = 'teacher_' + load_key
                if ('pts' in load_key) or ('img' in load_key):
                    dict_load = {_key.replace(load_key+'.',''):ckpt_load[_key] 
                            for _key in ckpt_load if load_key in _key}
                    getattr(self, teacher_key).load_state_dict(dict_load, strict=False)
                    print("Loaded pretrained {}".format(teacher_key))
                    assert len(dict_load) > 0

            if 'input_proj' in self.load_teacher:
                dict_load = {_key.replace('input_proj.',''):ckpt_load[_key] 
                            for _key in ckpt_load if 'input_proj' in _key}
                self.teacher_input_proj.load_state_dict(dict_load, strict=False)
                print("Loaded pretrained teacher_neck")
                assert len(dict_load) > 0

            if 'depth_head' in self.load_teacher:
                dict_load = {_key.replace('depth_net.',''):ckpt_load[_key] 
                            for _key in ckpt_load if 'depth_net' in _key}
                self.teacher_depth.load_state_dict(dict_load, strict=False)
                print("Loaded pretrained teacher_depth")
                assert len(dict_load) > 0

            if 'view_trans' in self.load_teacher:
                dict_load = {_key.replace('pts_bbox_head.view_trans.',''):ckpt_load[_key] 
                            for _key in ckpt_load if 'pts_bbox_head.view_trans' in _key}
                self.teacher_trans.load_state_dict(dict_load, strict=False)
                print("Loaded pretrained teacher_trans")
                assert len(dict_load) > 0
            
            if 'unified_conv' in self.load_teacher:
                dict_load = {_key.replace('pts_bbox_head.',''):ckpt_load[_key] 
                            for _key in ckpt_load if 'pts_bbox_head.conv_trans_head' in _key}
                self.load_state_dict(dict_load, strict=False)
                print("Loaded pretrained unified_conv")
                assert len(dict_load) > 0

    @property
    def with_depth_head(self):
        """bool: Whether the detector has a depth head."""
        return hasattr(self, 'depth_head') and self.depth_head is not None

    def process_pts(self, pts):
        if pts is None:
            return None, None
        
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        return voxel_features, coors

    @force_fp32(apply_to=('voxel_features', 'coors'))
    def extract_pts_feat(self, voxel_features, coors, pts_middle_encoder=None, pts_backbone=None, pts_neck=None):
        """Extract features of points.""" 
        batch_size = int(coors[-1, 0] + 1)
        x = pts_middle_encoder(voxel_features, coors, batch_size)
        if pts_backbone is not None:
            x = pts_backbone(x)
        if pts_neck is not None:
            x = pts_neck(x)
        return x


    def preprocess_img(self, img, img_metas):
        if img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
        
        return img, img_metas


    def extract_img_feat(self, img, batch, img_backbone, img_neck, input_proj):
        """Extract features of images."""
        if img is not None:
            img_feats = img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if img_neck is not None:
            img_feats = img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = input_proj(img_feat)
            img_feat = img_feat.view(batch, -1, *img_feat.shape[-3:])
            img_feats_reshaped.append(img_feat)
        return img_feats_reshaped


    @auto_fp16(apply_to=('img'))
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        pts_feats, img_feats, img_depth = None, None, None
        unified_feats_teacher = None
        batch = 1 if img is None else len(img)
        img, img_metas = self.preprocess_img(img, img_metas)
        voxel_feat, coors = self.process_pts(points)

        if self.distill_type in ['Multi_2D','Multi_3D'] and self.training:
            with torch.no_grad():
                # extract image features for teacher
                img_teacher = self.extract_img_feat(img, batch, 
                                    img_backbone=self.teacher_img_backbone, 
                                    img_neck=self.teacher_img_neck,
                                    input_proj=self.teacher_input_proj)
                depth_teacher = self.pred_depth(img_teacher, depth_net=self.teacher_depth)
                img_feats_teacher = self.teacher_trans(img_teacher, img_metas=img_metas, img_depth=depth_teacher)
                if len(img_feats_teacher.shape) == 6:
                    img_feats_teacher = img_feats_teacher.squeeze(1)
                
                # extract pts features for teacher
                pts_feats_teacher = self.extract_pts_feat(voxel_feat, coors,
                                                        pts_middle_encoder=self.teacher_pts_middle_encoder, 
                                                        pts_backbone=self.teacher_pts_backbone, 
                                                        pts_neck=self.teacher_pts_neck)

                out_before_relu = []
                unified_feats_teacher = img_feats_teacher + pts_feats_teacher
                for layer in self.unified_conv:
                    out_mid = layer[:-1](unified_feats_teacher)
                    out_before_relu.append(out_mid.clone())
                    unified_feats_teacher = layer[-1](out_mid)

            unified_feats_teacher = {'final':unified_feats_teacher, 'before_relu':out_before_relu}

            for _key in unified_feats_teacher:
                if isinstance(unified_feats_teacher[_key], list):
                    unified_feats_teacher[_key] = [_feat.detach() for _feat in unified_feats_teacher[_key]]
                else:
                    unified_feats_teacher[_key] = unified_feats_teacher[_key].detach()

        
        if self.distill_type in ['Multi_2D']:
            img_feats = self.extract_img_feat(img, batch, 
                                img_backbone=self.img_backbone, 
                                img_neck=self.img_neck,
                                input_proj=self.input_proj)
            img_depth = self.pred_depth(img_feats, depth_net=self.depth_net)
        
            return unified_feats_teacher, img_feats, img_depth
        elif self.distill_type in ['Multi_3D']:
            pts_feats = self.extract_pts_feat(voxel_feat, coors,
                                            pts_middle_encoder=self.pts_middle_encoder, 
                                            pts_backbone=self.pts_backbone, 
                                            pts_neck=self.pts_neck)
            return pts_feats, unified_feats_teacher, img_depth
        else:
            raise NotImplementedError

    @auto_fp16(apply_to=('img'))
    def pred_depth(self, img_feats, depth_net):
        if img_feats is None:
            return None
        
        if self.depth_head.type == "SimpleDepth":
            depth = []
            for _feat in img_feats:
                _depth = depth_net(_feat.view(-1,*_feat.shape[-3:]))
                _depth = _depth.softmax(dim=1)
                depth.append(_depth)
        else:
            raise NotImplementedError
        return depth


    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          depth_prob,
                          depth_coord,
                          img_metas,
                          img_depth=None,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas, img_depth)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, depth_prob, depth_coord, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      depth_prob=None,
                      depth_coord=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        pts_feat, img_feats, img_depth = self.extract_feat(points=points, img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(pts_feat, img_feats, 
                                            gt_bboxes_3d, gt_labels_3d, 
                                            depth_prob, depth_coord,
                                            img_metas, img_depth, 
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        return losses
    
    def forward_test(self, img_metas, points=None, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_metas)
        if points is not None:
            if num_augs != len(points):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(points), len(img_metas)))

        if num_augs == 1:
            if not isinstance(img_metas[0], list):
                img_metas = [img_metas]
            if not isinstance(img, list):
                img = [img]
            results = self.simple_test(img_metas[0], points, img[0], **kwargs)
        else:
            results = self.aug_test(points, img_metas, img, **kwargs)

        return results

    def simple_test_pts(self, pts_feat, img_feats, img_metas, img_depth, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(pts_feat, img_feats, img_metas, img_depth)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, points=None, img=None, rescale=False):
        """Test function without augmentaiton."""
        pts_feat, img_feats, img_depth = self.extract_feat(points=points, img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            pts_feat, img_feats, img_metas, img_depth, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        pts_feats, img_feats, img_depths = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_feats, img_depths, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        if points is None:
            points = [None] * len(img_metas)
        pts_feats, img_feats, img_depths = multi_apply(self.extract_feat, points, imgs,
                                                       img_metas)
        return pts_feats, img_feats, img_depths

    def aug_test_pts(self, pts_feats, img_feats, img_depths, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for _idx, img_meta in enumerate(img_metas):
            outs = self.pts_bbox_head(pts_feats[_idx], img_feats[_idx], 
                                      img_meta, img_depths[_idx])
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_meta, rescale=rescale)

            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_all_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)
        return merged_bboxes