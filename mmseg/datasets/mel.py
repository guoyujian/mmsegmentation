# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MelDataset(CustomDataset):


    CLASSES = ('bg', 'lesion')

    PALETTE = [[0, 0, 0], [255, 255, 255]]
    # def __init__(self, 
    #     pipeline, 
    #     img_dir, 
    #     img_suffix='.jpg', 
    #     ann_dir=None, 
    #     seg_map_suffix='.png', 
    #     split=None, 
    #     data_root=None, 
    #     test_mode=False, 
    #     ignore_index=10,  # 255
    #     reduce_zero_label=False, 
    #     gt_seg_map_loader_cfg=None, 
    #     file_client_args=dict(backend='disk')):

    #     super().__init__(pipeline, 
    #         img_dir, 
    #         img_suffix, 
    #         ann_dir, 
    #         seg_map_suffix, 
    #         split, 
    #         data_root, 
    #         test_mode, 
    #         ignore_index, 
    #         reduce_zero_label, 
    #         gt_seg_map_loader_cfg, 
    #         file_client_args,
    #         classes =  ('bg', 'lesion'), 
    #         palette = [[0, 0, 0], [255, 255, 255]])
    #     assert osp.exists(self.img_dir)


    def __init__(self, **kwargs):
        super(MelDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_segmentation.png',
            reduce_zero_label=False,
            ignore_index=10,
            classes= ('bg', 'lesion'),
            palette=[[0, 0, 0], [255, 255, 255]],
            **kwargs)
        assert osp.exists(self.img_dir)



        

