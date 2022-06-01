from re import I
import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadMultiViewMultiSweepImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, sweep_num=1, random_sweep=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.sweep_num = sweep_num
        self.random_sweep = random_sweep

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        results['filename'] = filename
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)

        img_sweeps = []
        sweeps_paths = results['cam_sweeps_paths']
        sweeps_ids = results['cam_sweeps_id']
        sweeps_time = results['cam_sweeps_time']
        if self.random_sweep:
            random_num = np.random.randint(0, self.sweep_num)
            sweeps_paths = [_sweep[:random_num] for _sweep in sweeps_paths]
            sweeps_ids = [_sweep[:random_num] for _sweep in sweeps_ids]
        else:
            random_num = self.sweep_num

        for _idx in range(len(sweeps_paths[0])):
            _sweep = np.stack(
                [mmcv.imread(name_list[_idx], self.color_type) for name_list in sweeps_paths], axis=-1)
            img_sweeps.append(_sweep)

        # add img sweeps to raw image
        img = np.stack([img, *img_sweeps], axis=-1)
        # img is of shape (h, w, c, num_views * sweep_num)
        img = img.reshape(*img.shape[:-2], -1)

        if self.to_float32:
            img = img.astype(np.float32)

        results['sweeps_paths'] = [[filename[_idx]] + sweeps_paths[_idx] for _idx in range(len(filename))]
        results['sweeps_ids'] = np.stack([[0]+_id for _id in sweeps_ids], axis=-1)
        results['sweeps_time'] = np.stack([[0]+_time for _time in sweeps_time], axis=-1)
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        # add sweep matrix to raw matrix
        results['lidar2img'] = [np.stack([results['lidar2img'][_idx], 
                                         *results['lidar2img_sweeps'][_idx][:random_num]], axis=0) 
                                         for _idx in range(len(results['lidar2img']))]
        results['lidar2cam'] = [np.stack([results['lidar2cam'][_idx], 
                                         *results['lidar2cam_sweeps'][_idx][:random_num]], axis=0) 
                                         for _idx in range(len(results['lidar2cam']))]
        results['cam_intrinsic'] = [np.stack([results['cam_intrinsic'][_idx], 
                                         *results['cam_sweeps_intrinsics'][_idx][:random_num]], axis=0) 
                                         for _idx in range(len(results['cam_intrinsic']))]
        results.pop('lidar2img_sweeps')
        results.pop('lidar2cam_sweeps')
        results.pop('cam_sweeps_intrinsics')

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str