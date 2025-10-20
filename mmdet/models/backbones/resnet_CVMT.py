# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath
import warnings
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from ..layers import ResLayer


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)
            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

@MODELS.register_module()
class ResNet50_CVMT(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs,)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3, #3
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super(ResNet50_CVMT, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)

            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.SFLLayers = nn.ModuleList([
        SFLLayer(dim=512, n_iter=1, sfeat_size=8, drop_path=0., layerscale=True,init_values=1.0e-5),
        SFLLayer(dim=1024, n_iter=1, sfeat_size=2, drop_path=0., layerscale=True,init_values=1.0e-5),
        SFLLayer(dim=2048, n_iter=1, sfeat_size=1, drop_path=0., layerscale=True,init_values=1.0e-5)
        ])

        self.HOFFLayers = nn.ModuleList([
            HOFFLayer(input_channels=512, k_order=5),
            HOFFLayer(input_channels=1024, k_order=2),
            HOFFLayer(input_channels=2048, k_order=1)
        ])

        self.DELayers = nn.ModuleList([
            DELayer(channels=512),
            DELayer(channels=1024),
            DELayer(channels=2048)
        ])
        self._freeze_stages()
        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        x_amp = x[0]
        x_ph =x[1]
        if self.deep_stem:
            x_amp = self.stem(x_amp)
            x_ph = self.stem(x_ph)
        else:
            x_amp = self.conv1(x_amp)
            x_ph = self.conv1(x_ph)

            x_amp = self.norm1(x_amp)
            x_ph = self.norm1(x_ph)

            x_amp = self.relu(x_amp)
            x_ph = self.relu(x_ph)

        x_amp = self.maxpool(x_amp)
        x_ph = self.maxpool(x_ph)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x_amp = res_layer(x_amp)
            x_ph =res_layer(x_ph)
            if i in self.out_indices:
                x_amp = self.SFLLayers[i-1](x_amp)
                x_ph = self.DELayers[i-1](x_ph)
                fused_feat = self.HOFFLayers[i-1](x_amp,x_ph)
                outs.append(fused_feat)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet50_CVMT, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


class SFLLayer(nn.Module):
    """
    A Super-Feature-Loop (SFL) block that:
      1) Applies a residual depthwise conv positional embedding (ResDWC).
      2) Normalizes and generates "super features" via iterative refinement.
      3) Optionally scales the update by a learnable gamma (LayerScale).
      4) Adds the (optionally scaled) residual back with DropPath for regularization.
    """

    def __init__(
        self,
        dim: int,
        n_iter: int,
        sfeat_size: int,
        drop_path: float = 0.0,
        layerscale: bool = True,
        init_values: float = 1.0e-5
    ):
        """
        Args:
            dim (int): Number of channels in the input feature map.
            n_iter (int): Number of refinement iterations in SuperFeatGeneration.
            sfeat_size (int): Spatial size for the generated super-features.
            drop_path (float): DropPath rate for stochastic depth. 0 means no DropPath.
            layerscale (bool): Whether to apply LayerScale (learnable scaling).
            init_values (float): Initial value for the LayerScale parameter gamma.
        """
        super().__init__()
        self.layerscale = layerscale

        # 1. Positional embedding via Residual Depthwise Convolution
        self.pos_embed = ResDWC(dim, kernel_size=3)

        # 2. Normalization before super-feature generation
        self.norm1 = LayerNorm2d(dim)

        # 3. Super-feature generation module with iterative updates
        self.sfg = SuperFeatGeneration(
            dim,
            sfeat_size=sfeat_size,
            n_iter=n_iter
        )
        # 4. DropPath for stochastic depth regularization (Identity if rate=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layerscale:
            # gamma_1 start very small to stabilize initial training
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(1, dim, 1, 1),
                requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SFL block.

        Steps:
          1) Add positional embedding.
          2) Normalize the result.
          3) Generate super-features and optionally scale via LayerScale.
          4) Apply DropPath and residual-add to the input.
          5) Return the updated features.

        Args:
            x (Tensor): Input feature map, shape (B, dim, H, W).

        Returns:
            Tensor: Output feature map with super-feature residual added.
        """
        # 1) Positional embedding
        x = self.pos_embed(x)

        # 2) Residual branch: normalize -> super-feature -> (scale) -> drop-path
        if self.layerscale:
            # apply LayerScale gamma before adding residual
            residual = self.gamma_1 * self.sfg(self.norm1(x))
        else:
            residual = self.sfg(self.norm1(x))

        # 3) Add residual (with DropPath) back into main path
        x = x + self.drop_path(residual)

        # Note: norm2 might be applied elsewhere if needed
        return x

class DELayer(nn.Module):
    """
    A depthwise separable convolution layer that performs a Laplacian-style
    high-frequency enhancement on the input features. Mathematically:
        out = x - LapConv(x)
    which emphasizes edges and fine details.
    """

    def __init__(self, channels: int):
        """
        Args:
            channels (int): Number of input and output channels.
        """
        super().__init__()
        # Learnable scaling factor for the Laplacian response
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

        # Depthwise convolution: each channel is convolved independently
        self.depthwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,  # groups=channels makes it depthwise
            bias=True
        )

        # Define a standard 3×3 Laplacian kernel:
        # [[ 0,  1,  0],
        #  [ 1, -4,  1],
        #  [ 0,  1,  0]]
        laplacian_kernel = torch.tensor(
            [[0., 1., 0.],
             [1., -4., 1.],
             [0., 1., 0.]],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 3, 3)

        # Initialize each channel's filter with the Laplacian kernel
        with torch.no_grad():
            for c in range(channels):
                # depthwise.weight shape is (channels, 1, 3, 3)
                self.depthwise.weight[c, 0].copy_(laplacian_kernel[0, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply depthwise Laplacian convolution and subtract
        it from the input to boost high-frequency components.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Enhanced feature map, same shape as input.
        """
        lap_out = self.depthwise(x)            # compute Laplacian response
        enhanced = x - self.alpha * lap_out    # subtract (scaled) response
        return enhanced
class HOFFLayer(nn.Module):
    """
    Higher-Order Fusion Feature (HOFF) block that:
      1) Applies k_order separate 1×1 convolutions on amplitude and phase branches.
      2) Performs a Laplacian-style high-frequency enhancement via Hadamard (element-wise) product
         between the corresponding amplitude and phase feature maps for each order.
      3) Optionally normalizes scalar weights via softmax.
      4) Produces the final fused output by weighted summation of all higher-order products.
    """

    def __init__(self, input_channels: int, k_order: int):
        """
        Args:
            input_channels (int): Number of channels in both amplitude and phase inputs.
            k_order (int): Number of higher-order convolution products to compute.
        """
        super().__init__()
        self.k_order = k_order
        self.use_softmax = True  # Whether to normalize weights with softmax

        # Learnable scalar weights for each order (initialized to 1.0)
        self.weights = nn.Parameter(torch.randn(k_order))
        nn.init.constant_(self.weights, 1.0)

        # Create k_order separate 1×1 conv layers for amplitude & phase branches
        self.kernels_amp = nn.ModuleList([
            nn.Conv2d(input_channels, input_channels, kernel_size=1)
            for _ in range(k_order)
        ])
        self.kernels_ph = nn.ModuleList([
            nn.Conv2d(input_channels, input_channels, kernel_size=1)
            for _ in range(k_order)
        ])

        # Initialize each conv layer with Kaiming uniform + zero bias
        for conv in list(self.kernels_amp) + list(self.kernels_ph):
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x_amp: torch.Tensor, x_ph: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to fuse amplitude and phase features.

        Args:
            x_amp (Tensor): Amplitude branch input of shape (B, C, H, W).
            x_ph  (Tensor): Phase branch input of shape (B, C, H, W).

        Returns:
            Tensor: Fused high-frequency features of shape (B, C, H, W).
        """
        mixed_list = []
        cumulative = []

        # 1) Compute element-wise product at each order
        for i in range(self.k_order):
            # 1×1 conv on amplitude branch
            amp_feat = self.kernels_amp[i](x_amp)
            # 1×1 conv on phase branch
            ph_feat = self.kernels_ph[i](x_ph)
            # Hadamard product for this order
            order_mix = amp_feat * ph_feat
            mixed_list.append(order_mix)

        # 2) Build cumulative products across orders
        fused = mixed_list[0]
        cumulative.append(fused)
        for i in range(1, self.k_order):
            # Multiply by next order feature map
            fused = fused * mixed_list[i]
            cumulative.append(fused)

        # 3) Normalize weights if enabled
        if self.use_softmax:
            weights = torch.softmax(self.weights, dim=0)  # shape (k_order,)
        else:
            weights = self.weights  # use raw learned scalars

        # 4) Expand weights to match feature dimensions
        # (k_order,) -> (k_order, 1, 1, 1, 1)
        expanded_weights = weights.view(-1, 1, 1, 1, 1)

        # 5) Stack cumulative features: shape (k_order, B, C, H, W)
        stacked = torch.stack(cumulative, dim=0)

        # 6) Weighted sum over orders: result shape (B, C, H, W)
        output = (expanded_weights * stacked).sum(dim=0)

        return output


class Mlp(nn.Module):
    """
    A convolutional MLP block: performs a pointwise Conv -> depthwise Conv -> activation -> dropout -> pointwise Conv -> dropout.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        # Set default hidden and output dims equal to input if not specified
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # 1×1 convolution to expand or project features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        # Depthwise convolution to mix spatial information
        self.dwconv = DWConv(hidden_features)
        # Non-linear activation
        self.act = act_layer()
        # Dropout for regularization
        self.drop = nn.Dropout(drop)
        # 1×1 convolution to project back to output dimension
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)     # pointwise conv
        x = self.dwconv(x)  # depthwise conv
        x = self.act(x)     # activation
        x = self.drop(x)    # dropout
        x = self.fc2(x)     # pointwise conv
        x = self.drop(x)    # dropout
        return x

class DWConv(nn.Module):
    """
    Depthwise convolution: a separate 3×3 conv per channel to mix spatial neighbors.
    """
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,  # groups=dim makes it depthwise
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)

class LayerNorm2d(nn.Module):
    """
    2D layer normalization applied per spatial position across channels.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Use standard LayerNorm on last dimension
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rearrange (B, C, H, W) -> (B, H, W, C), normalize, then back
        return self.norm(x.permute(0, 2, 3, 1).contiguous()) \
                   .permute(0, 3, 1, 2).contiguous()

class ResDWC(nn.Module):
    """
    Residual depthwise convolution: x -> x + DWC(x) to introduce positional bias.
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        # Depthwise convolution with padding to preserve spatial size
        self.conv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim  # depthwise
        )
        # Move conv to GPU if available
        device = torch.device('cuda')
        self.conv.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual addition of depthwise response
        return x + self.conv(x)

class BasicLayer(nn.Module):
    """
    A sequence of SFLLayer blocks, optionally with checkpointing.
    """
    def __init__(
        self,
        num_layers: int,
        dim: list,
        n_iter: int,
        stoken_size: int,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        layerscale: bool = False,
        init_values: float = 1.0e-5,
        downsample: bool = False,
        use_checkpoint: bool = False,
        checkpoint_num=None
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num

        # Build a list of SFLLayer blocks
        self.blocks = nn.ModuleList([
            SFLLayer(
                dim=dim[0],
                n_iter=n_iter,
                sfeat_size=stoken_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                layerscale=layerscale,
                init_values=init_values
            ) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequentially apply each block
        for blk in self.blocks:
            x = blk(x)
        return x

class SuperFeatGeneration(nn.Module):
    """
    Generate "super-pixel" features via iterative affinity-based pooling and refinement.

    Steps:
      1) Pool and reshape input into supertoken grid of size (hh, ww).
      2) Compute a pixel-to-supertoken affinity matrix via iterative soft assignments.
      3) Reconstruct refined supertoken features and optionally refine with conv layers.
      4) Project features back to pixel space.
    """
    def __init__(
        self,
        dim: int,
        sfeat_size: int,
        n_iter: int = 1,
        refine: bool = True,
        refine_attention: bool = False
    ):
        super().__init__()
        self.n_iter = n_iter
        self.sfeat_size = sfeat_size
        self.refine = refine
        self.refine_attention = refine_attention
        # scaling factor for similarity computation
        self.scale = dim ** -0.5

        # helper modules for patch extraction and projection
        self.unfold = Unfold(kernel_size=3)
        self.fold = Fold(kernel_size=3)

        # optional refinement convs for supertokens
        if refine:
            # both branches use same conv sequence here; can swap for attention version
            self.stoken_refine = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim),
                nn.Conv2d(dim, dim, kernel_size=1)
            )

    def sfeat_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Core forward pass to compute supertokens and reconstruct pixel features.

        Args:
            x: Input feature map (B, C, H, W).
        Returns:
            pixel_features: Refined output of shape (B, C, H, W).
        """
        B, C, H0, W0 = x.shape
        h = w = self.sfeat_size
        # compute padding to make H, W divisible by sfeat_size
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r or pad_b:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        _, _, H, W = x.shape

        # grid size for supertokens
        hh, ww = H // h, W // w
        # initial supertoken features via adaptive avg pool
        stokens = F.adaptive_avg_pool2d(x, (hh, ww))  # (B, C, hh, ww)
        # reshape pixel features: (B, C, hh, h, ww, w) -> (B, hh*ww, h*w, C)
        pixel_feats = x.view(B, C, hh, h, ww, w)
        pixel_feats = pixel_feats.permute(0, 2, 4, 3, 5, 1)
        pixel_feats = pixel_feats.reshape(B, hh*ww, h*w, C)

        affinity = None
        # iterative affinity refinement
        for it in range(self.n_iter):
            # extract local patches from supertokens: shape (B, hh*ww, C, 9)
            patches = self.unfold(stokens)          # (B, C*9, hh*ww)
            patches = patches.transpose(1, 2).reshape(B, hh*ww, C, 9)
            # compute pixel-to-stoken affinity via dot product
            affinity = (pixel_feats @ patches) * self.scale  # (B, hh*ww, h*w, 9)
            affinity = affinity.softmax(dim=-1)               # normalize over 9 neighbors

            # compute sum of affinities per patch, then fold to map
            affinity_sum = affinity.sum(dim=2)                # (B, hh*ww, 9)
            affinity_sum = affinity_sum.transpose(1,2).reshape(B, 9, hh, ww)
            affinity_sum = self.fold(affinity_sum)            # back to spatial map

            if it < self.n_iter - 1:
                # reconstruct stokens for next iteration
                stokens = pixel_feats.transpose(-1,-2) @ affinity  # (B, hh*ww, C, 9)
                stokens = stokens.reshape(B*C, 9, hh, ww)
                stokens = self.fold(stokens)
                stokens = stokens.reshape(B, C, hh, ww)
                # normalize by affinity sum
                stokens = stokens / (affinity_sum + 1e-12)

        # final supertoken reconstruction
        stokens = pixel_feats.transpose(-1,-2) @ affinity
        stokens = stokens.reshape(B*C, 9, hh, ww)
        stokens = self.fold(stokens).reshape(B, C, hh, ww)
        stokens = stokens / (affinity_sum.detach() + 1e-12)

        # optional refinement convs
        if self.refine:
            stokens = self.stoken_refine(stokens)

        # project stokens back to pixels
        stokens_patches = self.unfold(stokens)
        stokens_patches = stokens_patches.transpose(1,2).reshape(B, hh*ww, C, 9)
        pixel_out = stokens_patches @ affinity.permute(0,1,3,2)
        pixel_out = pixel_out.reshape(B, hh, ww, C, h, w)
        pixel_out = pixel_out.permute(0,3,1,4,2,5).reshape(B, C, H, W)

        # remove padding if applied
        if pad_r or pad_b:
            pixel_out = pixel_out[:,:,:H0,:W0]
        return pixel_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias forward to main stoken-based transform."""
        return self.sfeat_forward(x)


class Unfold(nn.Module):
    """
    Custom unfold that extracts 3×3 patches via a fixed eye-weighted conv.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        # identity patches weights: each center patch picks unique element
        weights = torch.eye(kernel_size**2).view(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        # convolve each channel separately, then reshape
        x = x.view(B*C,1,H,W)
        x = F.conv2d(x, self.weights, padding=self.kernel_size//2)
        return x.view(B, C*9, H*W)

class Fold(nn.Module):
    """
    Custom fold that reconstructs 3×3 patches via a fixed transposed conv.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        weights = torch.eye(kernel_size**2).view(kernel_size**2,1,kernel_size,kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,_,H,W = x.shape
        # transposed conv to map patches back to spatial map
        return F.conv_transpose2d(x, self.weights, padding=self.kernel_size//2)


