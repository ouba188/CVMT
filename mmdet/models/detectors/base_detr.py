from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class DetectionTransformer(BaseDetector, metaclass=ABCMeta):
    """
    Base class for Detection Transformer models.

    A Detection Transformer uses a Transformer encoder to process features
    from the neck, and a set of queries that interact with encoded features
    in a decoder. The head then performs bounding box regression and classification.

    Args:
        backbone (ConfigDict|dict): Config for the backbone network.
        neck (ConfigDict|dict, optional): Config for the neck. Defaults to None.
        encoder (ConfigDict|dict, optional): Config for the Transformer encoder. Defaults to None.
        decoder (ConfigDict|dict, optional): Config for the Transformer decoder. Defaults to None.
        bbox_head (ConfigDict|dict, optional): Config for the bbox head. Defaults to None.
        positional_encoding (ConfigDict|dict, optional): Positional encoding config. Defaults to None.
        num_queries (int): Number of queries in the decoder. Defaults to 100.
        train_cfg (ConfigDict|dict, optional): Training config for the bbox head. Defaults to None.
        test_cfg (ConfigDict|dict, optional): Testing config for the bbox head. Defaults to None.
        data_preprocessor (ConfigDict|dict, optional): Data pre-processing config (e.g. padding, mean, std). Defaults to None.
        init_cfg (ConfigDict|list, optional): Weight initialization config. Defaults to None.
    """
    def __init__(
        self,
        backbone: ConfigType,
        neck: OptConfigType = None,
        encoder: OptConfigType = None,
        decoder: OptConfigType = None,
        bbox_head: OptConfigType = None,
        positional_encoding: OptConfigType = None,
        num_queries: int = 100,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()

    @abstractmethod
    def _init_layers(self) -> None:
        """Initialize layers other than backbone, neck, and bbox head."""
        pass

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList
    ) -> Union[dict, list]:
        """
        Compute losses given a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images, shape (bs, channels, H, W).
            batch_data_samples (List[DetDataSample]): Ground truth and meta info.

        Returns:
            dict or list: Loss components.
        """

        img_feats = self.extract_feat(batch_inputs)
        head_inputs = self.forward_transformer(img_feats, batch_data_samples)
        losses = self.bbox_head.loss(**head_inputs, batch_data_samples=batch_data_samples)
        return losses

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True
    ) -> SampleList:
        """
        Run inference with post-processing on a batch of inputs.

        Args:
            batch_inputs (Tensor): Input images, shape (bs, channels, H, W).
            batch_data_samples (List[DetDataSample]): Meta info.
            rescale (bool): Whether to rescale boxes to original image size.

        Returns:
            List[DetDataSample]: Predictions added to data samples.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs = self.forward_transformer(img_feats, batch_data_samples)
        preds = self.bbox_head.predict(**head_inputs, rescale=rescale, batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, preds)
        return batch_data_samples

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList = None
    ) -> Tuple[List[Tensor]]:
        """
        Basic forward for training or feature extraction without post-processing.

        Args:
            batch_inputs (Tensor): Input images.
            batch_data_samples (List[DetDataSample], optional): Meta info.

        Returns:
            Tuple of Tensor: Raw outputs from bbox head forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs = self.forward_transformer(img_feats, batch_data_samples)
        outputs = self.bbox_head.forward(**head_inputs)
        return outputs

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None
    ) -> Dict:
        """
        Run the full Transformer pipeline: pre-transformer, encoder, pre-decoder, decoder.

        Args:
            img_feats (Tuple[Tensor]): Features from neck.
            batch_data_samples (List[DetDataSample], optional): Meta info.

        Returns:
            Dict: Inputs for bbox head, including hidden states and references.
        """
        enc_inputs, dec_inputs = self.pre_transformer(img_feats, batch_data_samples)
        enc_outputs = self.forward_encoder(**enc_inputs)
        tmp_dec_inputs, head_inputs = self.pre_decoder(**enc_outputs)
        dec_inputs.update(tmp_dec_inputs)
        dec_outputs = self.forward_decoder(**dec_inputs)
        head_inputs.update(dec_outputs)
        return head_inputs

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """
        Extract features using backbone and optional neck.

        Args:
            batch_inputs (Tensor): Input images.

        Returns:
            Tuple[Tensor]: Feature maps for Transformer.
        """
        x = self.backbone(batch_inputs)
        if getattr(self, 'with_neck', False):
            x = self.neck(x)
        return x

    @abstractmethod
    def pre_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None
    ) -> Tuple[Dict, Dict]:
        """
        Process image features and data samples before Transformer.

        Returns:
            (encoder_inputs, decoder_inputs).
        """
        pass

    @abstractmethod
    def forward_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        **kwargs
    ) -> Dict:
        """
        Run Transformer encoder on flattened features.

        Returns:
            Dict containing encoder outputs, e.g., memory.
        """
        pass

    @abstractmethod
    def pre_decoder(
        self,
        memory: Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Prepare queries, reference points and masks before decoder.

        Returns:
            (decoder_inputs, head_inputs).
        """
        pass

    @abstractmethod
    def forward_decoder(
        self,
        query: Tensor,
        query_pos: Tensor,
        memory: Tensor,
        **kwargs
    ) -> Dict:
        """
        Run Transformer decoder to refine queries.

        Returns:
            Dict with hidden states and references.
        """
        pass