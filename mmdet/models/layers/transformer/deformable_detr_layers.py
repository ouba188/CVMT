# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union  # 导入可选、元组和联合类型，用于类型提示

import torch  # 导入PyTorch主库
from mmcv.cnn import build_norm_layer  # 从mmcv中导入构建归一化层的函数
from mmcv.cnn.bricks.transformer import FFN, \
    MultiheadAttention  # 从mmcv的Transformer模块中导入前馈网络（FFN）和多头注意力（MultiheadAttention）
from mmcv.ops import MultiScaleDeformableAttention  # 从mmcv.ops中导入多尺度可变形注意力模块
from mmengine.model import ModuleList  # 从mmengine.model中导入ModuleList，用于存储子模块列表
from torch import Tensor, nn  # 从torch中直接导入Tensor类型和神经网络模块nn

from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)
# 从当前目录下的detr_layers模块中导入DETR相关的Transformer解码器、解码器层、编码器和编码器层

from .utils import inverse_sigmoid  # 从当前目录下的utils模块中导入inverse_sigmoid函数，用于逆sigmoid操作

# 尝试导入fairscale中的checkpoint_wrapper，用于梯度检查点以节省显存
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None  # 如果导入失败，则将checkpoint_wrapper置为None


class DeformableDetrTransformerEncoder(DetrTransformerEncoder):
    """Deformable DETR的Transformer编码器。

    该编码器继承自DetrTransformerEncoder，并在层的初始化和前向传播中使用
    多尺度可变形注意力模块来处理输入特征。
    """

    def _init_layers(self) -> None:
        """初始化编码器的各个层。"""
        # 利用ModuleList构造一个包含num_layers个编码器层的列表，每个层使用DeformableDetrTransformerEncoderLayer并传入配置
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        # 如果需要使用梯度检查点来降低GPU内存占用，则对前num_cp个层进行包装
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                # 如果checkpoint_wrapper未能成功导入，则抛出错误提示安装fairscale
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, '
                    'please install fairscale by executing the '
                    'following command: pip install fairscale.')
            for i in range(self.num_cp):
                # 对前num_cp个层使用checkpoint_wrapper包装，节省显存
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        # 将编码器的嵌入维度设置为第一个层的嵌入维度
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                **kwargs) -> Tensor:
        """Transformer编码器的前向传播函数。

        参数：
            query (Tensor): 输入的查询，形状为 (bs, num_queries, dim)。
            query_pos (Tensor): 查询的位置信息编码，形状为 (bs, num_queries, dim)。
            key_padding_mask (Tensor): 自注意力中用来屏蔽填充部分的mask，形状为 (bs, num_queries)。
            spatial_shapes (Tensor): 所有尺度特征的空间尺寸，形状为 (num_levels, 2)，最后一个维度为 (h, w)。
            level_start_index (Tensor): 每个尺度在展平特征中的起始索引，形状为 (num_levels, )，例如 [0, h_0*w_0, h_0*w_0+h_1*w_1, ...]。
            valid_ratios (Tensor): 各尺度中有效区域的比例，形状为 (bs, num_levels, 2)。
            **kwargs: 其他可能传入的额外参数。

        返回：
            Tensor: Transformer编码器的输出查询（也称为内存），形状为 (bs, num_queries, dim)。
        """
        # 获取编码器参考点，参考点的计算依赖于空间形状、有效比例和查询所在设备
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        # 遍历每一层，将前一层的输出作为下一层的输入进行处理
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
        # 返回最终经过所有层处理后的query
        return query

    @staticmethod
    def get_encoder_reference_points(
            spatial_shapes: Tensor, valid_ratios: Tensor,
            device: Union[torch.device, str]) -> Tensor:
        """获取编码器中使用的参考点。

        参数：
            spatial_shapes (Tensor): 各尺度的空间尺寸，形状为 (num_levels, 2)，最后一个维度表示 (h, w)。
            valid_ratios (Tensor): 各尺度有效区域的比例，形状为 (bs, num_levels, 2)。
            device (torch.device or str): 用于生成参考点的设备信息。

        返回：
            Tensor: 用于编码器的参考点，形状为 (bs, length, num_levels, 2)。
        """
        reference_points_list = []  # 初始化存储各尺度参考点的列表
        # 遍历每个尺度的空间尺寸
        for lvl, (H, W) in enumerate(spatial_shapes):
            # 为当前尺度生成y轴和x轴的网格坐标，坐标从0.5到H-0.5或W-0.5，共有H或W个点
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            # 将ref_y展平成一维，并增加一个batch维度，然后除以 (valid_ratio*H) 进行归一化
            ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
            # 同理处理x轴
            ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
            # 将x轴和y轴参考点沿最后一个维度堆叠，得到形状为 (bs, num_points, 2) 的参考点
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)  # 添加当前尺度的参考点到列表中
        # 将所有尺度的参考点在维度1上拼接起来
        reference_points = torch.cat(reference_points_list, 1)
        # 将参考点扩展一个维度，然后乘以valid_ratios进行归一化处理
        # 最终形状为 (bs, sum(hw), num_levels, 2)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points  # 返回计算得到的参考点


class DeformableDetrTransformerDecoder(DetrTransformerDecoder):
    """Deformable DETR的Transformer解码器。"""

    def _init_layers(self) -> None:
        """初始化解码器的各个层。"""
        # 构造一个ModuleList，包含num_layers个解码器层，每个层使用DeformableDetrTransformerDecoderLayer
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        # 将解码器的嵌入维度设置为第一个层的嵌入维度
        self.embed_dims = self.layers[0].embed_dims
        # 如果post_norm配置不为空，则抛出错误，因为Deformable DETR不使用后归一化
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Transformer解码器的前向传播函数。

        参数：
            query (Tensor): 输入查询，形状为 (bs, num_queries, dim)。
            query_pos (Tensor): 查询的位置编码，形状为 (bs, num_queries, dim)，在前向传播前将与query相加。
            value (Tensor): 输入的value，形状为 (bs, num_value, dim)。
            key_padding_mask (Tensor): cross_attn中用于屏蔽填充部分的mask，形状为 (bs, num_value)。
            reference_points (Tensor): 初始参考点，当as_two_stage为True时形状为 (bs, num_queries, 4)（排列为 (cx, cy, w, h)），否则为 (bs, num_queries, 2)（排列为 (cx, cy)）。
            spatial_shapes (Tensor): 各尺度的空间尺寸，形状为 (num_levels, 2)，最后一维为 (h, w)。
            level_start_index (Tensor): 每个尺度在展平特征中的起始索引，形如 [0, h_0*w_0, h_0*w_0+h_1*w_1, ...]。
            valid_ratios (Tensor): 各尺度有效区域的比例，形状为 (bs, num_levels, 2)。
            reg_branches (Optional[nn.Module]): 用于回归分支进行边界框修正，仅在with_box_refine为True时传入，否则为None。
            **kwargs: 其他额外参数。

        返回：
            tuple[Tensor]: 返回解码器的输出。
                - output (Tensor): 最后一层解码器输出的嵌入，如果return_intermediate为False，形状为 (num_queries, bs, embed_dims)，否则为 (num_decoder_layers, num_queries, bs, embed_dims)。
                - reference_points (Tensor): 最后一层解码器的参考点，如果return_intermediate为False，形状为 (bs, num_queries, 4)，否则为 (num_decoder_layers, bs, num_queries, 4)，排列为 (cx, cy, w, h)。
        """
        output = query  # 将初始查询赋值给output
        intermediate = []  # 用于存储中间各层的输出（如果需要返回中间结果）
        intermediate_reference_points = []  # 用于存储中间各层更新后的参考点
        # 遍历每个解码器层及其索引
        for layer_id, layer in enumerate(self.layers):
            # 判断参考点最后一个维度，如果为4（即包含宽高信息）
            if reference_points.shape[-1] == 4:
                # 构造参考点输入：将参考点扩展一维后乘以拼接后的valid_ratios（对宽高进行两次valid_ratios拼接）
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                # 如果参考点最后维度为2（仅包含(cx, cy)）
                assert reference_points.shape[-1] == 2
                # 参考点输入直接将参考点扩展一维后乘以valid_ratios
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]
            # 当前层进行前向传播，传入输出、查询位置编码、value、mask、空间形状、层起始索引、有效比例和参考点输入
            output = layer(
                output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)
            # 如果提供了回归分支，用于对输出进行边界框回归修正
            if reg_branches is not None:
                # 对当前层的输出进行回归预测，得到边界框偏移
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    # 当参考点包含宽高信息时，利用inverse_sigmoid进行逆变换，然后加上预测偏移，最后再进行sigmoid
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    # 当参考点仅包含(cx, cy)时，前两维进行逆sigmoid加预测偏移，其他维度保持预测值，然后sigmoid激活
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                                                    ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                # 更新参考点为新的预测值，并使用detach阻断梯度传播
                reference_points = new_reference_points.detach()
            # 如果设置返回中间结果，则保存当前层输出和参考点
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        # 如果返回中间结果，则将所有中间输出堆叠后返回
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        # 否则，返回最后一层的输出和参考点
        return output, reference_points


class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Deformable DETR的Transformer编码器层。"""

    def _init_layers(self) -> None:
        """初始化当前编码器层中的self_attn、ffn和归一化层。"""
        # 使用多尺度可变形注意力作为当前层的自注意力模块，并传入相应配置
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        # 将当前层的嵌入维度设置为self_attn的嵌入维度
        self.embed_dims = self.self_attn.embed_dims
        # 初始化前馈网络（FFN），传入对应配置
        self.ffn = FFN(**self.ffn_cfg)
        # 构造两个归一化层，使用build_norm_layer构造，每个归一化层的参数由norm_cfg和embed_dims决定
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        # 使用ModuleList存储归一化层
        self.norms = ModuleList(norms_list)


class DeformableDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Deformable DETR的Transformer解码器层。"""

    def _init_layers(self) -> None:
        """初始化当前解码器层中的自注意力、交叉注意力、ffn和归一化层。"""
        # 初始化自注意力模块，使用标准多头注意力，并传入自注意力配置
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        # 初始化交叉注意力模块，使用多尺度可变形注意力，并传入交叉注意力配置
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        # 将当前层的嵌入维度设置为自注意力模块的嵌入维度
        self.embed_dims = self.self_attn.embed_dims
        # 初始化前馈网络（FFN），传入对应配置
        self.ffn = FFN(**self.ffn_cfg)
        # 构造三个归一化层，使用build_norm_layer生成，每个归一化层使用相同配置和嵌入维度
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        # 使用ModuleList存储三个归一化层
        self.norms = ModuleList(norms_list)

# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Optional, Tuple, Union
#
# import torch
# from mmcv.cnn import build_norm_layer
# from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
# from mmcv.ops import MultiScaleDeformableAttention
# from mmengine.model import ModuleList
# from torch import Tensor, nn
#
# from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
#                           DetrTransformerEncoder, DetrTransformerEncoderLayer)
# from .utils import inverse_sigmoid
#
# try:
#     from fairscale.nn.checkpoint import checkpoint_wrapper
# except Exception:
#     checkpoint_wrapper = None
#
#
# class DeformableDetrTransformerEncoder(DetrTransformerEncoder):
#     """Transformer encoder of Deformable DETR."""
#
#     def _init_layers(self) -> None:
#         """Initialize encoder layers."""
#         self.layers = ModuleList([
#             DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
#             for _ in range(self.num_layers)
#         ])
#
#         if self.num_cp > 0:
#             if checkpoint_wrapper is None:
#                 raise NotImplementedError(
#                     'If you want to reduce GPU memory usage, \
#                     please install fairscale by executing the \
#                     following command: pip install fairscale.')
#             for i in range(self.num_cp):
#                 self.layers[i] = checkpoint_wrapper(self.layers[i])
#
#         self.embed_dims = self.layers[0].embed_dims
#
#     def forward(self, query: Tensor, query_pos: Tensor,
#                 key_padding_mask: Tensor, spatial_shapes: Tensor,
#                 level_start_index: Tensor, valid_ratios: Tensor,
#                 **kwargs) -> Tensor:
#         """Forward function of Transformer encoder.
#
#         Args:
#             query (Tensor): The input query, has shape (bs, num_queries, dim).
#             query_pos (Tensor): The positional encoding for query, has shape
#                 (bs, num_queries, dim).
#             key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
#                 input. ByteTensor, has shape (bs, num_queries).
#             spatial_shapes (Tensor): Spatial shapes of features in all levels,
#                 has shape (num_levels, 2), last dimension represents (h, w).
#             level_start_index (Tensor): The start index of each level.
#                 A tensor has shape (num_levels, ) and can be represented
#                 as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
#             valid_ratios (Tensor): The ratios of the valid width and the valid
#                 height relative to the width and the height of features in all
#                 levels, has shape (bs, num_levels, 2).
#
#         Returns:
#             Tensor: Output queries of Transformer encoder, which is also
#             called 'encoder output embeddings' or 'memory', has shape
#             (bs, num_queries, dim)
#         """
#         reference_points = self.get_encoder_reference_points(
#             spatial_shapes, valid_ratios, device=query.device)
#         for layer in self.layers:
#             query = layer(
#                 query=query,
#                 query_pos=query_pos,
#                 key_padding_mask=key_padding_mask,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 valid_ratios=valid_ratios,
#                 reference_points=reference_points,
#                 **kwargs)
#         return query
#
#     @staticmethod
#     def get_encoder_reference_points(
#             spatial_shapes: Tensor, valid_ratios: Tensor,
#             device: Union[torch.device, str]) -> Tensor:
#         """Get the reference points used in encoder.
#
#         Args:
#             spatial_shapes (Tensor): Spatial shapes of features in all levels,
#                 has shape (num_levels, 2), last dimension represents (h, w).
#             valid_ratios (Tensor): The ratios of the valid width and the valid
#                 height relative to the width and the height of features in all
#                 levels, has shape (bs, num_levels, 2).
#             device (obj:`device` or str): The device acquired by the
#                 `reference_points`.
#
#         Returns:
#             Tensor: Reference points used in decoder, has shape (bs, length,
#             num_levels, 2).
#         """
#
#         reference_points_list = []
#         for lvl, (H, W) in enumerate(spatial_shapes):
#             ref_y, ref_x = torch.meshgrid(
#                 torch.linspace(
#                     0.5, H - 0.5, H, dtype=torch.float32, device=device),
#                 torch.linspace(
#                     0.5, W - 0.5, W, dtype=torch.float32, device=device))
#             ref_y = ref_y.reshape(-1)[None] / (
#                 valid_ratios[:, None, lvl, 1] * H)
#             ref_x = ref_x.reshape(-1)[None] / (
#                 valid_ratios[:, None, lvl, 0] * W)
#             ref = torch.stack((ref_x, ref_y), -1)
#             reference_points_list.append(ref)
#         reference_points = torch.cat(reference_points_list, 1)
#         # [bs, sum(hw), num_level, 2]
#         reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#         return reference_points
#
#
# class DeformableDetrTransformerDecoder(DetrTransformerDecoder):
#     """Transformer Decoder of Deformable DETR."""
#
#     def _init_layers(self) -> None:
#         """Initialize decoder layers."""
#         self.layers = ModuleList([
#             DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
#             for _ in range(self.num_layers)
#         ])
#         self.embed_dims = self.layers[0].embed_dims
#         if self.post_norm_cfg is not None:
#             raise ValueError('There is not post_norm in '
#                              f'{self._get_name()}')
#
#     def forward(self,
#                 query: Tensor,
#                 query_pos: Tensor,
#                 value: Tensor,
#                 key_padding_mask: Tensor,
#                 reference_points: Tensor,
#                 spatial_shapes: Tensor,
#                 level_start_index: Tensor,
#                 valid_ratios: Tensor,
#                 reg_branches: Optional[nn.Module] = None,
#                 **kwargs) -> Tuple[Tensor]:
#         """Forward function of Transformer decoder.
#
#         Args:
#             query (Tensor): The input queries, has shape (bs, num_queries,
#                 dim).
#             query_pos (Tensor): The input positional query, has shape
#                 (bs, num_queries, dim). It will be added to `query` before
#                 forward function.
#             value (Tensor): The input values, has shape (bs, num_value, dim).
#             key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
#                 input. ByteTensor, has shape (bs, num_value).
#             reference_points (Tensor): The initial reference, has shape
#                 (bs, num_queries, 4) with the last dimension arranged as
#                 (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
#                 shape (bs, num_queries, 2) with the last dimension arranged
#                 as (cx, cy).
#             spatial_shapes (Tensor): Spatial shapes of features in all levels,
#                 has shape (num_levels, 2), last dimension represents (h, w).
#             level_start_index (Tensor): The start index of each level.
#                 A tensor has shape (num_levels, ) and can be represented
#                 as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
#             valid_ratios (Tensor): The ratios of the valid width and the valid
#                 height relative to the width and the height of features in all
#                 levels, has shape (bs, num_levels, 2).
#             reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
#                 the regression results. Only would be passed when
#                 `with_box_refine` is `True`, otherwise would be `None`.
#
#         Returns:
#             tuple[Tensor]: Outputs of Deformable Transformer Decoder.
#
#             - output (Tensor): Output embeddings of the last decoder, has
#               shape (num_queries, bs, embed_dims) when `return_intermediate`
#               is `False`. Otherwise, Intermediate output embeddings of all
#               decoder layers, has shape (num_decoder_layers, num_queries, bs,
#               embed_dims).
#             - reference_points (Tensor): The reference of the last decoder
#               layer, has shape (bs, num_queries, 4)  when `return_intermediate`
#               is `False`. Otherwise, Intermediate references of all decoder
#               layers, has shape (num_decoder_layers, bs, num_queries, 4). The
#               coordinates are arranged as (cx, cy, w, h)
#         """
#         output = query
#         intermediate = []
#         intermediate_reference_points = []
#         for layer_id, layer in enumerate(self.layers):
#             if reference_points.shape[-1] == 4:
#                 reference_points_input = \
#                     reference_points[:, :, None] * \
#                     torch.cat([valid_ratios, valid_ratios], -1)[:, None]
#             else:
#                 assert reference_points.shape[-1] == 2
#                 reference_points_input = \
#                     reference_points[:, :, None] * \
#                     valid_ratios[:, None]
#             output = layer(
#                 output,
#                 query_pos=query_pos,
#                 value=value,
#                 key_padding_mask=key_padding_mask,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 valid_ratios=valid_ratios,
#                 reference_points=reference_points_input,
#                 **kwargs)
#
#             if reg_branches is not None:
#                 tmp_reg_preds = reg_branches[layer_id](output)
#                 if reference_points.shape[-1] == 4:
#                     new_reference_points = tmp_reg_preds + inverse_sigmoid(
#                         reference_points)
#                     new_reference_points = new_reference_points.sigmoid()
#                 else:
#                     assert reference_points.shape[-1] == 2
#                     new_reference_points = tmp_reg_preds
#                     new_reference_points[..., :2] = tmp_reg_preds[
#                         ..., :2] + inverse_sigmoid(reference_points)
#                     new_reference_points = new_reference_points.sigmoid()
#                 reference_points = new_reference_points.detach()
#
#             if self.return_intermediate:
#                 intermediate.append(output)
#                 intermediate_reference_points.append(reference_points)
#
#         if self.return_intermediate:
#             return torch.stack(intermediate), torch.stack(
#                 intermediate_reference_points)
#
#         return output, reference_points
#
#
# class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
#     """Encoder layer of Deformable DETR."""
#
#     def _init_layers(self) -> None:
#         """Initialize self_attn, ffn, and norms."""
#         self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
#         self.embed_dims = self.self_attn.embed_dims
#         self.ffn = FFN(**self.ffn_cfg)
#         norms_list = [
#             build_norm_layer(self.norm_cfg, self.embed_dims)[1]
#             for _ in range(2)
#         ]
#         self.norms = ModuleList(norms_list)
#
#
# class DeformableDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
#     """Decoder layer of Deformable DETR."""
#
#     def _init_layers(self) -> None:
#         """Initialize self_attn, cross-attn, ffn, and norms."""
#         self.self_attn = MultiheadAttention(**self.self_attn_cfg)
#         self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
#         self.embed_dims = self.self_attn.embed_dims
#         self.ffn = FFN(**self.ffn_cfg)
#         norms_list = [
#             build_norm_layer(self.norm_cfg, self.embed_dims)[1]
#             for _ in range(3)
#         ]
#         self.norms = ModuleList(norms_list)
