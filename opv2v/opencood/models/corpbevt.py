"""
Implementation of Brady Zhou's cross view transformer
"""
import einops
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix

# --- DA-BEV MODIFICATION START ---
from opencood.utils.da_bev_utils import SpatialDomainDiscriminator
# --- DA-BEV MODIFICATION END ---


class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, spatial_correction_matrix):
        """
        Transform the bev features to ego space.

        Parameters
        ----------
        x : torch.Tensor
            B L C H W
        spatial_correction_matrix : torch.Tensor
            Transformation matrix to ego

        Returns
        -------
        The bev feature same shape as x but with transformation
        """
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)

        # transpose and flip to make the transformation correct
        x = rearrange(x, 'b l c h w  -> b l c w h')
        x = torch.flip(x, dims=(4,))
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)

        # flip and transpose back
        x = cav_features
        x = torch.flip(x, dims=(4,))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x


class CorpBEVT(nn.Module):
    def __init__(self, config):
        super(CorpBEVT, self).__init__()
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        fax_params = config['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

        if config['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(128, config['compression'])
        else:
            self.compression = False

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])

        # spatial fusion
        self.fusion_net = SwapFusionEncoder(config['fax_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])

        # --- DA-BEV MODIFICATION START ---
        # Note: You can override these in your cobevt.yaml by adding `uda_iv_channels` 
        # and `uda_bev_channels` to the top level of the config file. 
        # Defaulting to 256 for both, which is standard for CoBEVT ResNet output and fusion output.
        iv_dim = config["uda_iv_channels"]
        
        # If naive compressor is used, the channel size drops to 128 before fusion.
        default_bev_dim = 128 if self.compression else 256
        bev_dim = config["uda_bev_channels"]
        
        self.iv_discriminator = SpatialDomainDiscriminator(in_channels=iv_dim)
        self.bev_discriminator = SpatialDomainDiscriminator(in_channels=bev_dim)
        # --- DA-BEV MODIFICATION END ---

    # --- DA-BEV MODIFICATION START ---
    # Changed signature to accept alpha and is_target parameters
    def forward(self, batch_dict, alpha=1.0, is_target=False):
    # --- DA-BEV MODIFICATION END ---
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        
        # --- DA-BEV MODIFICATION START ---
        # ResnetEncoder returns a list of multi-scale features. We take the last one.
        img_features = x[-1] if isinstance(x, (list, tuple)) else x
        
        # img_features shape is [B, L, M, C, H, W]. We must flatten to 4D for Conv2D
        B, L, M, C_dim, H_img, W_img = img_features.shape
        img_features_4d = img_features.view(B * L * M, C_dim, H_img, W_img)
        
        # Forward through discriminator: outputs [B*L*M, 1]
        domain_pred_iv_flat = self.iv_discriminator(img_features_4d, alpha)
        
        # Reshape and average across agents/cameras to get [B, 1] 
        # This aligns the batch size with the BEV predictions for the QAL loss calculation
        domain_pred_iv = domain_pred_iv_flat.view(B, L * M, -1).mean(dim=1)
        # --- DA-BEV MODIFICATION END ---

        x = self.fax(batch_dict)

        # B*L, C, H, W
        x = x.squeeze(1)

        # compressor
        if self.compression:
            x = self.naive_compressor(x)

        # Reformat to (B, max_cav, C, H, W)
        x, mask = regroup(x, record_len, self.max_cav)
        # perform feature spatial transformation,  B, max_cav, H, W, C
        x = self.sttf(x, transformation_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(x.shape,
                                      mask,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)

        # fuse all agents together to get a single bev map, b h w c
        x = rearrange(x, 'b l h w c -> b l c h w')
        x = self.fusion_net(x, com_mask)
        
        # --- DA-BEV MODIFICATION START ---
        domain_pred_bev = self.bev_discriminator(x, alpha)
        
        # If this is the target domain, we do not need to calculate the segmentation loss.
        # Bypass the decoder entirely to save GPU memory and compute.
        if is_target:
            return {
                'domain_pred_iv': domain_pred_iv,
                'domain_pred_bev': domain_pred_bev
            }
        # --- DA-BEV MODIFICATION END ---

        x = x.unsqueeze(1)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        # --- DA-BEV MODIFICATION START ---
        output_dict.update({
            'domain_pred_iv': domain_pred_iv,
            'domain_pred_bev': domain_pred_bev
        })
        # --- DA-BEV MODIFICATION END ---

        return output_dict