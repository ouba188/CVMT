# Copyright (c) OpenMMLab. All rights reserved.
import math  # Import math module for mathematical operations (e.g., constant π, logarithms)
from typing import Dict, Tuple  # Import Dict and Tuple from typing module for type hints

import torch  # Import PyTorch main module
import torch.nn.functional as F  # Import PyTorch neural network functions module (activation, convolution, etc.)
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention  # Import multi-scale deformable attention module from mmcv
from mmengine.model import xavier_init  # Import Xavier initialization function from mmengine
from torch import Tensor, nn  # Import Tensor type and neural network module nn from torch
from torch.nn.init import normal_  # Import normal distribution initialization function from torch.nn.init

from mmdet.registry import MODELS  # Import MODELS registry from mmdet for model registration
from mmdet.structures import OptSampleList  # Import OptSampleList type from mmdet (optional data sample list)
from mmdet.utils import OptConfigType  # Import OptConfigType type from mmdet (optional configuration)
from ..layers import (DeformableDetrTransformerDecoder,
                      DeformableDetrTransformerEncoder, SinePositionalEncoding)  # Import layers from parent directory
from .base_detr import DetectionTransformer  # Import DetectionTransformer base class from current directory


@MODELS.register_module()  # Register the following class to mmdet's MODELS registry
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
                 decoder: OptConfigType = None,  # Transformer decoder configuration
                 bbox_head: OptConfigType = None,  # Bounding box head configuration
                 with_box_refine: bool = False,  # Flag for box refinement
                 as_two_stage: bool = False,  # Flag for two-stage structure
                 num_feature_levels: int = 4,  # Number of multi-scale feature levels
                 **kwargs) -> None:
        self.with_box_refine = with_box_refine  # Store box refinement flag
        self.as_two_stage = as_two_stage  # Store two-stage structure flag
        self.num_feature_levels = num_feature_levels  # Store number of feature levels

        if bbox_head is not None:  # Check if bbox_head configuration is provided
            assert 'share_pred_layer' not in bbox_head and \
                   'num_pred_layer' not in bbox_head and \
                   'as_two_stage' not in bbox_head, \
                'The keyword arguments `share_pred_layer`, `num_pred_layer`, and `as_two_stage` are set in `detector.__init__()`, ' \
                'and should not be specified in the `bbox_head` configuration.'
            # When using two-stage structure, the last prediction layer generates proposals from encoder features
            # When box refinement is enabled, all prediction layers share parameters
            bbox_head['share_pred_layer'] = not with_box_refine  # Set share_pred_layer based on box refinement
            bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1) \
                if self.as_two_stage else decoder['num_layers']  # Set number of prediction layers
            bbox_head['as_two_stage'] = as_two_stage  # Pass as_two_stage flag to bbox_head

        super().__init__(*args, decoder=decoder, bbox_head=bbox_head, **kwargs)  # Call parent class initialization

    def _init_layers(self) -> None:
        """Initialize layers excluding backbone, neck, and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)  # Initialize sine positional encoding module
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)  # Initialize Transformer encoder
        self.decoder = DeformableDetrTransformerDecoder(**self.decoder)  # Initialize Transformer decoder
        self.embed_dims = self.encoder.embed_dims  # Store embedding dimension from encoder
        if not self.as_two_stage:
            # For non-two-stage structure, initialize query embedding (split into query and query_pos later)
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)

        num_feats = self.positional_encoding.num_feats  # Get number of positional features
        assert num_feats * 2 == self.embed_dims, \
            f'Embedding dimension should be exactly twice the number of positional features. ' \
            f'Found {self.embed_dims} and {num_feats}.'  # Validate embedding dimension

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))  # Initialize level embedding parameters

        if self.as_two_stage:
            # For two-stage structure, initialize memory and position transformation layers
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)  # Memory transformation FC layer
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)  # Memory transformation LayerNorm
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)  # Position transformation FC layer
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)  # Position transformation LayerNorm
        else:
            # For non-two-stage structure, initialize reference points FC layer
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()  # Call parent class weight initialization
        for coder in self.encoder, self.decoder:  # Initialize parameters for encoder and decoder
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)  # Xavier uniform initialization for multi-dimensional parameters
        for m in self.modules():  # Initialize MultiScaleDeformableAttention modules
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            # Initialize weights for memory and position transformation layers in two-stage structure
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            # Initialize reference points FC layer with Xavier uniform distribution
            xavier_init(self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embed)  # Normal initialization for level embedding

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
        batch_size = mlvl_feats[0].size(0)  # Get batch size from the first level feature

        # Construct binary masks for the Transformer
        assert batch_data_samples is not None  # Ensure batch data samples are provided
        batch_input_shape = batch_data_samples[0].batch_input_shape  # Get batch input shape
        input_img_h, input_img_w = batch_input_shape  # Extract image height and width
        img_shape_list = [sample.batch_input_shape for sample in batch_data_samples]  # Get shapes of all images in the batch
        same_shape_flag = all([s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list])  # Check if all images have the same shape

        # Handle ONNX export or uniform image shapes (no mask needed)
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)  # No mask for each level
                mlvl_pos_embeds.append(self.positional_encoding(None, input=feat))  # Compute positional encoding
        else:
            # For non-uniform image shapes, construct masks
            masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))  # Initialize masks with all ones
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]  # Get actual image size for each sample
                masks[img_id, :img_h, :img_w] = 0  # Set valid regions to 0

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                # Interpolate mask to match feature map size and convert to boolean
                mlvl_masks.append(F.interpolate(masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))  # Compute positional encoding with mask

        # Flatten features, positional embeddings, and masks across levels
        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape  # Get feature shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)  # Get spatial shape (h, w)
            # Flatten feature from (bs, c, h, w) to (bs, h*w, c)
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            # Flatten positional embedding similarly
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            # Add level embedding to positional embedding
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # Flatten mask from (bs, h, w) to (bs, h*w)
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # Concatenate across levels
        feat_flatten = torch.cat(feat_flatten, 1)  # (bs, num_feat_points, dim)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # (bs, num_feat_points, dim)
        mask_flatten = torch.cat(mask_flatten, 1) if mask_flatten[0] is not None else None  # (bs, num_feat_points)

        # Process spatial shapes and level start indices
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)  # (num_levels, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # (num_levels,)

        # Compute valid ratios for each level
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1) if mlvl_masks[0] is not None else \
            mlvl_feats[0].new_ones(batch_size, len(mlvl_feats), 2)  # (bs, num_levels, 2)

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
        batch_size, _, c = memory.shape  # Get batch size and embedding dimension

        if self.as_two_stage:
            # Generate proposals from encoder outputs for two-stage structure
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](output_memory)  # Class predictions
            enc_outputs_coord_unact = self.bbox_head.reg_branches[self.decoder.num_layers](output_memory) + output_proposals  # Unactivated coordinate predictions
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()  # Activated coordinates (0-1 normalized)

            # Select top-k proposals based on foreground scores
            topk_proposals = torch.topk(enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()  # Detach to prevent gradient propagation
            reference_points = topk_coords_unact.sigmoid()  # Activated reference points

            # Transform proposal position embeddings
            pos_trans_out = self.pos_trans_fc(self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)
            query_pos, query = torch.split(pos_trans_out, c, dim=2)  # Split into query and query_pos
        else:
            # For non-two-stage structure, use query embeddings
            enc_outputs_class, enc_outputs_coord = None, None  # No encoder outputs for proposals
            query_embed = self.query_embedding.weight  # Get query embeddings
            query_pos, query = torch.split(query_embed, c, dim=1)  # Split into query and query_pos
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)  # Expand to batch dimension
            query = query.unsqueeze(0).expand(batch_size, -1, -1)  # Expand to batch dimension
            reference_points = self.reference_points_fc(query_pos).sigmoid()  # Generate reference points

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()  # Only include during training
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
            reg_branches=self.bbox_head.reg_branches if self.with_box_refine else None)  # Pass regression branches if refining boxes
        references = [reference_points, *inter_references]  # Combine initial and intermediate references
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
        _, H, W = mask.shape  # Get feature map height and width
        valid_H = torch.sum(~mask[:, :, 0], 1)  # Valid height (non-masked rows)
        valid_W = torch.sum(~mask[:, 0, :], 1)  # Valid width (non-masked columns)
        valid_ratio_h = valid_H.float() / H  # Height ratio
        valid_ratio_w = valid_W.float() / W  # Width ratio
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # Stack into (bs, 2)
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
        bs = memory.size(0)  # Get batch size
        proposals = []  # List to store proposals from each level
        _cur = 0  # Current start index in the flattened feature sequence

        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW  # Height and width of current level features

            if memory_mask is not None:
                # Extract mask for current level and compute valid regions
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)  # Scaling factor for coordinates
            else:
                # Use feature map size directly if no mask
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)  # (w, h) scaling factor

            # Generate grid coordinates (cx, cy)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # (H, W, 2)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale  # Normalize coordinates (0-1)

            # Set box width/height (scaled by level)
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)  # Width/height increases with level
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)  # (bs, H*W, 4) [cx, cy, w, h]
            proposals.append(proposal)
            _cur += (H * W)  # Update start index for next level

        output_proposals = torch.cat(proposals, 1)  # Concatenate proposals across levels

        # Filter valid proposals (coordinates within 0.01-0.99)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).sum(-1, keepdim=True) == output_proposals.shape[-1]
        # Inverse sigmoid transformation
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # Mask invalid proposals
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        # Transform memory (mask invalid regions and apply FC + LayerNorm)
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
        scale = 2 * math.pi  # Scaling factor
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)  # Frequency indices
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)  # Temperature scaling

        proposals = proposals.sigmoid() * scale  # Normalize and scale proposals
        pos = proposals[:, :, :, None] / dim_t  # (bs, num_queries, 4, num_pos_feats)
        # Apply sin/cos to even/odd dimensions and flatten
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos