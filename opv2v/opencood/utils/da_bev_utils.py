import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class SpatialDomainDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(SpatialDomainDiscriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(max(1, in_channels // 4), 1)

    def forward(self, x, alpha):
        x = grad_reverse(x, alpha)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return torch.sigmoid(logits)

def compute_da_bev_qal_loss(src_iv_pred, tgt_iv_pred, src_bev_pred, tgt_bev_pred):
    bce = nn.BCELoss(reduction='none')
    
    src_labels = torch.ones_like(src_iv_pred)
    tgt_labels = torch.zeros_like(tgt_iv_pred)
    
    loss_iv_src = bce(src_iv_pred, src_labels)
    loss_iv_tgt = bce(tgt_iv_pred, tgt_labels)
    loss_bev_src = bce(src_bev_pred, src_labels)
    loss_bev_tgt = bce(tgt_bev_pred, tgt_labels)
    
    with torch.no_grad():
        lambda_iv = torch.exp(-(torch.log(src_iv_pred + 1e-6) + torch.log(1 - tgt_iv_pred + 1e-6)))
        lambda_bev = torch.exp(-(torch.log(src_bev_pred + 1e-6) + torch.log(1 - tgt_bev_pred + 1e-6)))
        lambda_iv = torch.clamp(lambda_iv, max=5.0)
        lambda_bev = torch.clamp(lambda_bev, max=5.0)

    loss_qal = torch.mean(lambda_bev * (loss_iv_src + loss_iv_tgt) + \
                          lambda_iv * (loss_bev_src + loss_bev_tgt))
    return loss_qal