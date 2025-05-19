# Copyright (c) OpenMMLab. All rights reserved.
#import torch
from mmengine.model import BaseModel
from mmdet.registry import MODELS
from mmdet.models import DINO


@MODELS.register_module()
class MultiBranchDINO(BaseModel):
    def __init__(self, amplitude_model_cfg, phase_model_cfg):
        super(MultiBranchDINO, self).__init__()

        # 构建幅度模型和相位模型
        self.amplitude_model = DINO(amplitude_model_cfg)
        self.phase_model = DINO(phase_model_cfg)

    def forward(self, amplitude_data, phase_data, **kwargs):
        # 前向传播
        amp_out = self.amplitude_model(amplitude_data)
        phase_out = self.phase_model(phase_data)

        return amp_out, phase_out

    def train_step(self, data, optimizer):
        amplitude_data, phase_data, targets = data

        # 训练幅度模型和相位模型
        amp_out = self.amplitude_model(amplitude_data)
        phase_out = self.phase_model(phase_data)

        # 计算损失
        loss = self.compute_loss(amp_out, phase_out, targets)

        # 更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def compute_loss(self, amp_out, phase_out, targets):
        # 自定义损失计算
        # 这里可以将幅度和相位的输出合并进行损失计算
        loss = 0
        # 示例：可以使用交叉熵损失或其他自定义损失
        # loss += some_loss_function(amp_out, targets)
        # loss += some_loss_function(phase_out, targets)
        return loss

    def forward_test(self, amplitude_data, phase_data, **kwargs):
        # 测试模式的前向传播
        amp_out = self.amplitude_model(amplitude_data, **kwargs)
        phase_out = self.phase_model(phase_data, **kwargs)

        return amp_out, phase_out

    def get_current_loss(self):
        # 如果需要，返回当前的损失
        return self.loss



