# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (DeformableDetrTransformerDecoder,
                      DeformableDetrTransformerEncoder, SinePositionalEncoding)
from .base_detr import DetectionTransformer


@MODELS.register_module()
class DeformableDETR(DetectionTransformer):
    r"""Implementation of the model from the paper `Deformable DETR: Deformable Transformers for
    End-to-End Object Detection <https://arxiv.org/abs/2010.04159>`_

    Code modified from the official GitHub repository
    <https://github.com/fundamentalvision/Deformable-DETR>.

    Args:
        decoder (:obj:`ConfigDict` or dict, optional): Configuration for the Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Configuration for the bounding box head module. Defaults to None.
        with_box_refine (bool, optional): Whether to iteratively refine reference boxes in the decoder. Defaults to False.
        as_two_stage (bool, optional): Whether to use a two-stage structure (generate proposals from encoder outputs). Defaults to False.
        num_feature_levels (int, optional): Number of feature levels. Defaults to 4.
    """

    def __init__(self,
                 *args,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 num_feature_levels: int = 4,
                 **kwargs) -> None:
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels

        if bbox_head is not None:
            assert 'share_pred_layer' not in bbox_head and \
                   'num_pred_layer' not in bbox_head and \
                   'as_two_stage' not in bbox_head, \
                'The keyword arguments `share_pred_layer`, `num_pred_layer`, and `as_two_stage` are set in `detector.__init__()`, ' \
                'and should not be specified in the `bbox_head` configuration.'

            bbox_head['share_pred_layer'] = not with_box_refine
            bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1) \
                if self.as_two_stage else decoder['num_layers']
            bbox_head['as_two_stage'] = as_two_stage

        super().__init__(*args, decoder=decoder, bbox_head=bbox_head, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers excluding backbone, neck, and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DeformableDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:

            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'Embedding dimension should be exactly twice the number of positional features. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:

            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:

            xavier_init(self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embed)

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the Transformer.

        Transformer forward workflow:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        See `TransformerDetector.forward_transformer` in `mmdet/detector/base_detr.py` for details.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features (output from neck), each with shape (bs, dim, h_lvl, w_lvl).
            batch_data_samples (list[:obj:`DetDataSample`], optional): Batch data samples containing image info (e.g., gt_instance).

        Returns:
            tuple[dict]: Two dictionaries containing inputs for encoder and decoder respectively.
                - encoder_inputs_dict (dict): Contains 'feat', 'feat_mask', 'feat_pos', 'spatial_shapes', 'level_start_index', 'valid_ratios'.
                - decoder_inputs_dict (dict): Contains 'memory_mask', 'spatial_shapes', 'level_start_index', 'valid_ratios'.
        """
        batch_size = mlvl_feats[0].size(0)
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.batch_input_shape for sample in batch_data_samples]
        same_shape_flag = all([s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list])


        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(self.positional_encoding(None, input=feat))
        else:

            masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                
                mlvl_masks.append(F.interpolate(masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))  

        
        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape  
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)  
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # Concatenate across levels
        feat_flatten = torch.cat(feat_flatten, 1)  
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  
        mask_flatten = torch.cat(mask_flatten, 1) if mask_flatten[0] is not None else None 

        # Process spatial shapes and level start indices
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2) 
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])) 

        # Compute valid ratios for each level
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1) if mlvl_masks[0] is not None else \
            mlvl_feats[0].new_ones(batch_size, len(mlvl_feats), 2)  

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)

        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward pass through the Transformer encoder.

        Transformer forward workflow:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        See `TransformerDetector.forward_transformer` in `mmdet/detector/base_detr.py` for details.

        Args:
            feat (Tensor): Flattened features, shape (bs, num_feat_points, dim).
            feat_mask (Tensor): Feature padding mask, shape (bs, num_feat_points).
            feat_pos (Tensor): Positional embeddings of features, shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features across levels, shape (num_levels, 2).
            level_start_index (Tensor): Start indices of each level in the flattened sequence, shape (num_levels,).
            valid_ratios (Tensor): Valid region ratios for each level, shape (bs, num_levels, 2).

        Returns:
            dict: Encoder outputs containing 'memory', 'memory_mask', and 'spatial_shapes'.
        """
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables (e.g., `query`, `query_pos`, `reference_points`) before entering the Transformer decoder.

        Transformer forward workflow:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        See `TransformerDetector.forward_transformer` in `mmdet/detector/base_detr.py` for details.

        Args:
            memory (Tensor): Encoder output memory, shape (bs, num_feat_points, dim).
            memory_mask (Tensor): Memory padding mask, shape (bs, num_feat_points) (used only when `as_two_stage=True`).
            spatial_shapes (Tensor): Spatial shapes of features across levels, shape (num_levels, 2) (used only when `as_two_stage=True`).

        Returns:
            tuple[dict, dict]:
                - decoder_inputs_dict (dict): Inputs for decoder (e.g., 'query', 'query_pos', 'memory', 'reference_points').
                - head_inputs_dict (dict): Inputs for bbox_head (empty during inference).
        """
        batch_size, _, c = memory.shape  

        if self.as_two_stage:
           
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](output_memory)  
            enc_outputs_coord_unact = self.bbox_head.reg_branches[self.decoder.num_layers](output_memory) + output_proposals 
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid() 
            topk_proposals = torch.topk(enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()  
            reference_points = topk_coords_unact.sigmoid()  
            pos_trans_out = self.pos_trans_fc(self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)
            query_pos, query = torch.split(pos_trans_out, c, dim=2)  
        else:
            enc_outputs_class, enc_outputs_coord = None, None  
            query_embed = self.query_embedding.weight  
            query_pos, query = torch.split(query_embed, c, dim=1) 
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1) 
            query = query.unsqueeze(0).expand(batch_size, -1, -1)  
            reference_points = self.reference_points_fc(query_pos).sigmoid() 

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()  
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, reference_points: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward pass through the Transformer decoder.

        Transformer forward workflow:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        See `TransformerDetector.forward_transformer` in `mmdet/detector/base_detr.py` for details.

        Args:
            query (Tensor): Decoder input queries, shape (bs, num_queries, dim).
            query_pos (Tensor): Positional embeddings of queries, shape (bs, num_queries, dim).
            memory (Tensor): Encoder output memory, shape (bs, num_feat_points, dim).
            memory_mask (Tensor): Memory padding mask, shape (bs, num_feat_points).
            reference_points (Tensor): Initial reference points, shape (bs, num_queries, 4) (two-stage) or (bs, num_queries, 2).
            spatial_shapes (Tensor): Spatial shapes of features across levels, shape (num_levels, 2).
            level_start_index (Tensor): Start indices of each level, shape (num_levels,).
            valid_ratios (Tensor): Valid region ratios, shape (bs, num_levels, 2).

        Returns:
            dict: Decoder outputs containing 'hidden_states' (decoder layer outputs) and 'references' (reference points).
        """
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches if self.with_box_refine else None)  
        references = [reference_points, *inter_references] 
        decoder_outputs_dict = dict(hidden_states=inter_states, references=references)
        return decoder_outputs_dict

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Calculate the valid region ratio for a feature map.

        For a mask (bs, H, W), valid ratios are computed as:
            r_h = valid_H / H, r_w = valid_W / W
        These ratios normalize image coordinates to feature map coordinates.

        Args:
            mask (Tensor): Binary mask of the feature map, shape (bs, H, W).

        Returns:
            Tensor: Valid ratios [r_w, r_h], shape (1, 2).
        """
        _, H, W = mask.shape  
        valid_H = torch.sum(~mask[:, :, 0], 1)  
        valid_W = torch.sum(~mask[:, 0, :], 1)  
        valid_ratio_h = valid_H.float() / H 
        valid_ratio_w = valid_W.float() / W  
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) 
        return valid_ratio

    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoder outputs (used only when `as_two_stage=True`).

        Args:
            memory (Tensor): Encoder output memory, shape (bs, num_feat_points, dim).
            memory_mask (Tensor): Memory padding mask, shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features across levels, shape (num_levels, 2).

        Returns:
            tuple[Tensor, Tensor]:
                - output_memory (Tensor): Transformed memory, shape (bs, num_feat_points, dim).
                - output_proposals (Tensor): Inverse-normalized proposals, shape (bs, num_keys, 4) (cx, cy, w, h).
        """
        bs = memory.size(0) 
        proposals = [] 
        _cur = 0  

        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW  

            if memory_mask is not None:
                
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)  
            else:
                
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)  

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale  
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)  
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)  
            proposals.append(proposal)
            _cur += (H * W) 

        output_proposals = torch.cat(proposals, 1)  
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).sum(-1, keepdim=True) == output_proposals.shape[-1]
        output_proposals = torch.log(output_proposals / (1 - output_proposals))

        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)

        return output_memory, output_proposals

    @staticmethod
    def get_proposal_pos_embed(proposals: Tensor,
                               num_pos_feats: int = 128,
                               temperature: int = 10000) -> Tensor:
        """Compute position embeddings for proposals.

        Args:
            proposals (Tensor): Non-normalized proposals, shape (bs, num_queries, 4) (cx, cy, w, h).
            num_pos_feats (int, optional): Feature dimension per coordinate axis. Defaults to 128.
            temperature (int, optional): Temperature for scaling embeddings. Defaults to 10000.

        Returns:
            Tensor: Position embeddings, shape (bs, num_queries, num_pos_feats * 4).
        """
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos
