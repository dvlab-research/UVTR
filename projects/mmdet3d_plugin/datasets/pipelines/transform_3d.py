import numpy as np
from numpy import random
import torch
import mmcv
import cv2
from mmdet.datasets.builder import PIPELINES
from mmcv.utils import build_from_cfg
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.builder import OBJECTSAMPLERS


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """
    Random scale the image
    """
    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        np.random.shuffle(self.scales)
        rand_scale = self.scales[0]
        if isinstance(results['img_shape'], list):
            img_shape = results['img_shape'][0]
        else:
            img_shape = results['img_shape']
        y_size = int(img_shape[0] * rand_scale)
        x_size = int(img_shape[1] * rand_scale) 
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size, y_size), return_scale=False) for img in results['img']]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        # NOT flip GT during training
        # results['gt_bboxes_3d'].tensor[:, :6] *= rand_scale
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class ImageRandomResizeCropFlip(object):
    def __init__(self, flip_ratio=None, resize_scales=None, crop_sizes=None, training=True):
        self.flip_ratio = flip_ratio
        self.resize_scales = resize_scales
        self.crop_sizes = crop_sizes
        self.training = training

    def _resize_img(self, results):
        H, W = results['img_shape'][0][:2] if isinstance(results['img_shape'], list) else results['img_shape'][:2]

        resize = np.random.uniform(*self.resize_scales)
        newW, newH = int(W * resize), int(H * resize)
        results['img'] = [mmcv.imresize(img, (newW, newH), return_scale=False) for img in results['img']]
        results['img_scale_mat'] = torch.Tensor([[resize, 0], [0, resize]])
        if isinstance(results['img_shape'], list):
            for _result in results['img_shape']:
                _result = (newH, newW, _result[-1])
        else:
            results['img_shape'] = (newH, newW, results['img_shape'][-1])
        return results

    def _flip_img(self, results, flip_type='horizontal'):
        if np.random.rand() >= self.flip_ratio or (not self.training):
            return results
        
        results['img'] = [mmcv.imflip(img, flip_type) for img in results['img']]
        img_width = results['img_shape'][0][1] if isinstance(results['img_shape'], list) else results['img_shape'][1]

        results['img_flip_rot'] = torch.Tensor([[-1, 0], [0, 1]])
        results['img_flip_trans'] = torch.Tensor([img_width, 0])
        return results

    def _crop_img(self, results):
        H, W = results['img_shape'][0][:2] if isinstance(results['img_shape'], list) else results['img_shape'][:2]
        # crop from image bottom
        start_h = H - self.crop_sizes[0]
        if self.training:
            start_w = int(np.random.uniform(0, max(0, W - self.crop_sizes[1])))
        else:
            start_w = max(0, W - self.crop_sizes[1]) // 2

        results['img'] = [img[start_h:start_h+self.crop_sizes[0], start_w:start_w+self.crop_sizes[1], ...] 
                            for img in results['img']]
        
        results['img_crop_trans'] = torch.Tensor([-start_w, -start_h])

        if isinstance(results['img_shape'], list):
            for _result in results['img_shape']:
                _result = (*self.crop_sizes, _result[-1])
        else:
            results['img_shape'] = (*self.crop_sizes, results['img_shape'][-1])
        return results

    def __call__(self, results):
        # resize image
        if self.resize_scales is not None:
            results = self._resize_img(results)
        # flip image
        if self.flip_ratio is not None:
            results = self._flip_img(results)
        # crop image
        if self.crop_sizes is not None:
            results = self._crop_img(results)

        results['img_rot_aug'] = torch.eye(2)
        results['img_trans_aug'] = torch.Tensor([0, 0])

        if 'img_scale_mat' in results:
            results['img_rot_aug'] = results['img_rot_aug'] @ results['img_scale_mat']

        if 'img_flip_rot' in results:
            results['img_rot_aug'] = results['img_rot_aug'] @ results['img_flip_rot']
            results['img_trans_aug'] += results['img_flip_trans']

        if 'img_crop_trans' in results:
            results['img_trans_aug'] += results['img_crop_trans']

        return results


@PIPELINES.register_module()
class UnifiedRotScaleTrans(object):
    """
    Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range
        self.shift_height = shift_height


    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if 'rot_degree' in input_dict:
            noise_rotation = input_dict['rot_degree']
        else:
            rotation = self.rot_range
            noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # calculate rotation matrix
        rot_sin = torch.sin(torch.tensor(noise_rotation))
        rot_cos = torch.cos(torch.tensor(noise_rotation))
        rot_mat_T = torch.Tensor([[rot_cos, -rot_sin, 0],
                                  [rot_sin, rot_cos, 0],
                                  [0, 0, 1]])
        input_dict['uni_rot_mat'] = rot_mat_T

        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T
            else:
                input_dict[key].rotate(noise_rotation)
                    

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        if 'points' in input_dict:
            points = input_dict['points']
            points.scale(scale)
            if self.shift_height:
                assert 'height' in points.attribute_dims.keys(), \
                    'setting shift_height=True but points have no height attribute'
                points.tensor[:, points.attribute_dims['height']] *= scale
            input_dict['points'] = points

        input_dict['uni_scale_mat'] = torch.Tensor([[scale, 0, 0], 
                                                    [0, scale, 0], 
                                                    [0, 0, scale]])

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        # unified augmentation for point and voxel
        uni_rot_aug = input_dict['uni_rot_mat'] @ input_dict['uni_scale_mat']
        if 'uni_rot_aug' in input_dict:
            uni_rot_aug = input_dict['uni_rot_aug'] @ uni_rot_aug
        input_dict['uni_rot_aug'] = uni_rot_aug

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str
    

@PIPELINES.register_module()
class UnifiedRandomFlip3D(object):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                flip_ratio_bev_horizontal=0.0,
                flip_ratio_bev_vertical=0.0,
                **kwargs):
        super(UnifiedRandomFlip3D, self).__init__()
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        if 'pcd_horizontal_flip' not in input_dict:
            flip_horizontal = True if np.random.rand(
            ) < self.flip_ratio_bev_horizontal else False
            input_dict['pcd_horizontal_flip'] = flip_horizontal
        if 'pcd_vertical_flip' not in input_dict:
            flip_vertical = True if np.random.rand(
            ) < self.flip_ratio_bev_vertical else False
            input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        # flips the y (horizontal) or x (vertical) axis
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
            flip_mat[1,1] *= -1
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
            flip_mat[0,0] *= -1

        # unified augmentation for point and voxel
        uni_rot_aug = flip_mat
        if 'uni_rot_aug' in input_dict:
            uni_rot_aug = input_dict['uni_rot_aug'] @ uni_rot_aug
        input_dict['uni_rot_aug'] = uni_rot_aug
        
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str
    

@PIPELINES.register_module()
class UnifiedObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, sample_method='depth', modify_points=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.sample_method = sample_method
        self.modify_points = modify_points
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                with_img=True)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, with_img=False)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_points_idx = sampled_dict["points_idx"]
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points_idx = -1 * np.ones(len(points), dtype=np.int)
            # check the points dimension
            # points = points.cat([sampled_points, points])
            points = points.cat([points, sampled_points])
            points_idx = np.concatenate([points_idx, sampled_points_idx], axis=0)

            if self.sample_2d:
                imgs = input_dict['img']
                lidar2img = input_dict['lidar2img']
                sampled_img = sampled_dict['images']
                sampled_num = len(sampled_gt_bboxes_3d)
                imgs, points_keep = self.unified_sample(imgs, lidar2img, 
                                            points.tensor.numpy(), 
                                            points_idx, gt_bboxes_3d.corners.numpy(), 
                                            sampled_img, sampled_num)
                
                input_dict['img'] = imgs

                if self.modify_points:
                    points = points[points_keep]

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        return input_dict

    def unified_sample(self, imgs, lidar2img, points, points_idx, bboxes_3d, sampled_img, sampled_num):
        # for boxes
        bboxes_3d = np.concatenate([bboxes_3d, np.ones_like(bboxes_3d[..., :1])], -1)
        is_raw = np.ones(len(bboxes_3d))
        is_raw[-sampled_num:] = 0
        is_raw = is_raw.astype(bool)
        raw_num = len(is_raw)-sampled_num
        # for point cloud
        points_3d = points[:,:4].copy()
        points_3d[:,-1] = 1
        points_keep = np.ones(len(points_3d)).astype(np.bool)
        new_imgs = imgs

        assert len(imgs)==len(lidar2img) and len(sampled_img)==sampled_num
        for _idx, (_img, _lidar2img) in enumerate(zip(imgs, lidar2img)):
            coord_img = bboxes_3d @ _lidar2img.T
            coord_img[...,:2] /= coord_img[...,2,None]
            depth = coord_img[...,2]
            img_mask = (depth > 0).all(axis=-1)
            img_count = img_mask.nonzero()[0]
            if img_mask.sum() == 0:
                continue
            depth = depth.mean(1)[img_mask]
            coord_img = coord_img[...,:2][img_mask]
            minxy = np.min(coord_img, axis=-2)
            maxxy = np.max(coord_img, axis=-2)
            bbox = np.concatenate([minxy, maxxy], axis=-1).astype(int)
            bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=_img.shape[1]-1)
            bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=_img.shape[0]-1)
            img_mask = ((bbox[:,2:]-bbox[:,:2]) > 1).all(axis=-1)
            if img_mask.sum() == 0:
                continue
            depth = depth[img_mask]
            if 'depth' in self.sample_method:
                paste_order = depth.argsort()
                paste_order = paste_order[::-1]
            else:
                paste_order = np.arange(len(depth), dtype=np.int64)
            img_count = img_count[img_mask][paste_order]
            bbox = bbox[img_mask][paste_order]

            paste_mask = -255 * np.ones(_img.shape[:2], dtype=np.int)
            fg_mask = np.zeros(_img.shape[:2], dtype=np.int)
            # first crop image from raw image
            raw_img = []
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    raw_img.append(_img[_box[1]:_box[3],_box[0]:_box[2]])

            # then stitch the crops to raw image
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    _img[_box[1]:_box[3],_box[0]:_box[2]] = raw_img.pop(0)
                    fg_mask[_box[1]:_box[3],_box[0]:_box[2]] = 1
                else:
                    img_crop = sampled_img[_count-raw_num]
                    if len(img_crop)==0: continue
                    img_crop = cv2.resize(img_crop, tuple(_box[[2,3]]-_box[[0,1]]))
                    _img[_box[1]:_box[3],_box[0]:_box[2]] = img_crop

                paste_mask[_box[1]:_box[3],_box[0]:_box[2]] = _count
            
            new_imgs[_idx] = _img

            # calculate modify mask
            if self.modify_points:
                points_img = points_3d @ _lidar2img.T
                points_img[:,:2] /= points_img[:,2,None]
                depth = points_img[:,2]
                img_mask = depth > 0
                if img_mask.sum() == 0:
                    continue
                img_mask = (points_img[:,0] > 0) & (points_img[:,0] < _img.shape[1]) & \
                           (points_img[:,1] > 0) & (points_img[:,1] < _img.shape[0]) & img_mask
                points_img = points_img[img_mask].astype(int)
                new_mask = paste_mask[points_img[:,1], points_img[:,0]]==(points_idx[img_mask]+raw_num)
                raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_num)
                raw_bg = (fg_mask == 0) & (paste_mask < 0)
                raw_mask = raw_fg[points_img[:,1], points_img[:,0]] | raw_bg[points_img[:,1], points_img[:,0]]
                keep_mask = new_mask | raw_mask
                points_keep[img_mask] = points_keep[img_mask] & keep_mask

        return new_imgs, points_keep

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str