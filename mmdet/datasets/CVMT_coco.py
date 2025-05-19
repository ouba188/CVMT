# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union
from mmengine.fileio import get_local_path
from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset

@DATASETS.register_module()
class CVMTCocoDataset(BaseDetDataset):
    """Dataset for CVMT COCO with amplitude and phase images."""
    METAINFO = {
        'classes':
        ('ship'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(0, 255, 0)]
    }
    COCOAPI = COCO
    ANN_ID_UNIQUE = True

    def __init__(self, data_prefix: dict = dict(img_prefix_amp='', img_prefix_phase=''), **kwargs): #kwargs是
        self.img_prefix_amp = data_prefix.get('img_prefix_amp', None)
        self.img_prefix_phase = data_prefix.get('img_prefix_phase', None)
        super().__init__(data_prefix=data_prefix, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file."""
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []

        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })

            # Add amplitude and phase image paths
            amp_img_path = osp.join(self.img_prefix_amp, raw_img_info['file_name'])
            phase_img_path = osp.join(self.img_prefix_phase, raw_img_info['file_name'])
            parsed_data_info['img_path_amp'] = amp_img_path
            parsed_data_info['img_path_phase'] = phase_img_path

            data_list.append(parsed_data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), (
                f"Annotation ids in '{self.ann_file}' are not unique!"
            )

        del self.coco
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format."""
        data_info = {}
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info['img_path_amp'] = osp.join(self.img_prefix_amp, img_info['file_name'])
        data_info['img_path_phase'] = osp.join(self.img_prefix_phase, img_info['file_name'])
        data_info['img_path'] = self.data_root+data_info['img_path_amp']
        data_info['img_id'] = img_info['img_id']
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        instances = []
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            instance = {
                'bbox': bbox,
                'bbox_label': self.cat2label[ann['category_id']],
                'ignore_flag': int(ann.get('iscrowd', False))
            }
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            instances.append(instance)

        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg."""
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for data_info in self.data_list:
            if data_info['img_id'] not in ids_in_cat:
                continue
            if filter_empty_gt and not data_info['instances']:
                continue
            if min_size > 0:
                valid = any(
                    inst['bbox'][2] > min_size and inst['bbox'][3] > min_size for inst in data_info['instances'])
                if not valid:
                    continue
            valid_data_infos.append(data_info)
        return valid_data_infos

    def full_init(self):
        "Load the comment file and set 'BaseDataset._fully_initialized' to True." ""
        if self._fully_initialized:
            return

        self.data_list = self.load_data_list()

        self.data_list = self.filter_data()

        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True