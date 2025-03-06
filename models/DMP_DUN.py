import math

import torch
from torch import nn
from timm.models.layers import to_2tuple
from imageCS_utils.utils import ImageEmbedding, ModelTools, PhiClass
from imageCS_utils.get_matrix import get_standard_orthogonal_matrix_torch_init
from imageCS_utils.base_utils.info import Info

from utils import create_model
from models.Guided_Diffusion import UNetModel

class MyUNetModel(nn.Module):
    def __init__(self, unet:UNetModel):
        super().__init__()

        self.dtype = unet.dtype

        self.time_embed = unet.time_embed
        self.model_channels = unet.model_channels

        self.input_blocks = unet.input_blocks
        self.middle_block = unet.middle_block
        self.output_blocks = unet.output_blocks
        self.out = unet.out

        self.rgb2y = nn.Conv2d(6, 1, kernel_size=1, stride=1)
        self.y2rgb = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        ModelTools.freeze_(self.input_blocks)
        ModelTools.freeze_(self.middle_block)
        ModelTools.freeze_(self.output_blocks)
        ModelTools.freeze_(self.out)
        ModelTools.freeze_(self.time_embed)

        self.max_period = 10000
        half = self.model_channels // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def timestep_embedding(self, timesteps):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """

        args = timesteps[:, None] * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.model_channels % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x:torch.Tensor, timesteps:torch.Tensor):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(self.timestep_embedding(timesteps))

        ###>>> fp32 to fp16
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        ###<<< fp16 to fp32

        x = self.out(h)

        return x
    
class MyRes(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        return x + self.model(x)

class MyResBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, res_num):
        super().__init__()

        self.enc = nn.Conv2d(in_channels, inner_channels, 3, 1, 1, bias=True)
        self.res = nn.Sequential(
            *[MyRes(inner_channels) for _ in range(res_num)]
        )
        self.dec = nn.Conv2d(inner_channels, out_channels, 3, 1, 1, bias=True)
    
    def forward(self, x):
        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)

        return x

class Diffusion(nn.Module):
    def __init__(self, model_kwargs, model_path, phi_size, sensing_rate, time_step, start_time=1000, img_dim=1, res_inner_channels=32, res_inner_num=2):
        super().__init__()
        #####> init <#####
        self.sensing_rate = sensing_rate
        (crop_h, crop_w) = self.phi_size = to_2tuple(phi_size)
        self.cs_in = crop_h * crop_w
        self.cs_out = int(self.cs_in * float(sensing_rate))
        self.img_dim = img_dim
        self._init_phi()

        #####> init unet model <#####
        unet:UNetModel = create_model(
            UNetModel=UNetModel,
            **model_kwargs
        ) # type: ignore
        if model_kwargs["use_fp16"]:
            unet.convert_to_fp16()
        #unet.dtype = torch.float32

        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            unet.load_state_dict(state_dict)
        except Exception as e:
            Info.WARN(f"Fail to load pretrained guided-diffusion weights: {str(e)}")

        #####> init my unet <#####
        self.my_unet = MyUNetModel(unet)
        self.enc = MyResBlock(
            in_channels = 3,
            inner_channels = res_inner_channels,
            out_channels = res_inner_channels,
            res_num = 4
        )

        #####> init diffusion parameters <#####
        self.T = 1000
        self.beta_1 = 0.0001
        self.beta_T = 0.02

        assert start_time <= self.T
        assert start_time % time_step == 0
        self.time_step = time_step
        self.start_time = start_time

        self._init_diffusion_params()

        #####> init other models <#####
        self.dec = MyResBlock(
            in_channels = 3,
            inner_channels = res_inner_channels,
            out_channels = self.img_dim,
            res_num = 4
        )

        inner_stage_num = self.timestep_list.size(0)
        self.res_list = nn.ParameterList([
            MyResBlock(
                in_channels = res_inner_channels,
                inner_channels = res_inner_channels,
                out_channels = res_inner_channels,
                res_num = res_inner_num
            ) for _ in range(inner_stage_num)
        ])

        self.grad_steps = [torch.ones((1, )) for _ in range(inner_stage_num)]

    def _init_phi(self):
        phi_matrix = get_standard_orthogonal_matrix_torch_init(self.cs_out, self.cs_in)
        self.phi = nn.Parameter(phi_matrix, requires_grad=True)
        self.phi_class = PhiClass(self.phi, self.phi_size)

    def _init_diffusion_params(self):
        # Init all diffusion param
        T = self.T
        beta_1 = self.beta_1
        beta_T = self.beta_T

        beta_tmp = torch.linspace(beta_1, beta_T, T).to(torch.float32)
        alpha_tmp = 1 - beta_tmp

        alpha = torch.cumprod(alpha_tmp, dim=0)
        alpha = torch.cat([torch.ones(size=(1,), dtype=torch.float32), alpha], dim=0)
        alpha = alpha.view(-1, 1, 1, 1)

        t = torch.arange(0, T+1, 1, dtype=torch.long)
        t = t.view(-1, 1)

        # Init used diffusion param
        time_step = self.time_step
        start_time = self.start_time

        x_coeff_list = []
        noise_pred_coeff_list = []
        t_now_list = []

        t_indexs = torch.arange(start_time, 0, -time_step, dtype=torch.long)
        for t_index in t_indexs:
            t_now = t[t_index]
            t_next = t_now - time_step
            (x_coeff, noise_pred_coeff, _) = self.get_coeff(
                alpha_list = alpha,
                t_now = t_now,
                t_next = t_next,
                with_noise = False
            )

            x_coeff_list.append(x_coeff)
            noise_pred_coeff_list.append(noise_pred_coeff)
            t_now_list.append(t_now)
        
        x_coeff_list = torch.stack(x_coeff_list, dim=0)
        noise_pred_coeff_list = torch.stack(noise_pred_coeff_list, dim=0)
        t_now_list = torch.stack(t_now_list, dim=0)

        self.x_coeff_list = nn.Parameter(x_coeff_list)
        self.noise_pred_coeff_list = nn.Parameter(noise_pred_coeff_list)
        self.timestep_list = nn.Parameter(t_now_list.float())

    def get_coeff(self, alpha_list, t_now, t_next, with_noise=False):
        r"""
        < DDIM process >
        if with_noise: sigma_t = 0

        x_0t = 
            x_t * (1 / alpha_t).sqrt() -
            noise_pred_t * ((1 - alpha_t) / alpha_t).sqrt()

        x_t_next = 
            x_0t * alpha_t_next.sqrt() +
            noise_pred_t * (1 - alpha_t_next - sigma_t**2).sqrt() +
            N(0, 1) * sigma_t
        
        """

        alpha = alpha_list[t_now]
        alpha_next = alpha_list[t_next]

        if with_noise:
            sigma = (1 - alpha_next) * (1 - alpha / alpha_next) / (1 - alpha)
            sigma = sigma.sqrt()
        else:
            sigma = 0

        xt_coeff = (alpha_next / alpha).sqrt()

        noise_pred_coeff = -(alpha_next * (1 - alpha) / alpha).sqrt()
        noise_pred_coeff += (1 - alpha_next - sigma**2).sqrt()

        norm_noise_coeff = sigma

        return (xt_coeff, noise_pred_coeff,  norm_noise_coeff)

    def reverse_diffusion_process_step(self, x_t, timestep, x_coeff, noise_pred_coeff):
        """
        GPU Memory may out, so split image into patches when not training
        x_t.shape: (B, C, H, W)
        t.shape: (B)
        """
        (bs, channels, img_h, img_w) = x_t.size()
        patch_size = 256

        if self.training:
            x_next = self._reverse_diffusion_process_step(x_t, timestep, x_coeff, noise_pred_coeff)
        else:
            x_patch = ImageEmbedding.image2patches(x_t, patch_size, embed_patch=False)
            x_next_list = []
            for xp_t in x_patch:
                xp_t = xp_t.unsqueeze(0)
                x_next = self._reverse_diffusion_process_step(xp_t, timestep, x_coeff, noise_pred_coeff)
                x_next_list.append(x_next)
            x_next = torch.cat(x_next_list, dim=0)
            x_next = ImageEmbedding.patches2image(x_next, (img_h, img_w), patch_size, embed_patch=False)

        return x_next

    def _reverse_diffusion_process_step(self, x_t, timestep, x_coeff, noise_pred_coeff):
        """
        x_t.shape: (B, C, H, W)
        t.shape: (B)
        """
        # Prepare
        noise_pred = self.my_unet(x_t, timestep)

        if noise_pred.size(1) == 6:
            noise_pred = noise_pred[:, :3, ...]

        # Formula
        x_next = x_coeff * x_t + noise_pred_coeff * noise_pred

        # Return
        return x_next
    
    def _3c_phi(self, x:torch.Tensor):
        """ 3-channels image, process phi channels by channels """
        x = (x + 1) / 2

        assert x.size(1) == 3
        b, c, h, w = x.size()
        x = x.view(b*c, 1, h, w)

        y = self.phi_class.phi(x)
        return y
    
    def _3c_phiT(self, y:torch.Tensor):
        """ 3-channels image, process phiT channels by channels """
        x = self.phi_class.phiT(y)
        b, c, h, w = x.size()
        x = x.view(-1, 3, h, w)

        x = x * 2 - 1
        return x
    
    def grad_descent(self, xt, y, grad_step):
        phiTphix = self._3c_phiT(self._3c_phi(xt))
        phiTy = self._3c_phiT(y)
        xt = xt - grad_step * (phiTphix + phiTy)

        return xt
    
    def _inv_img(self, img:torch.Tensor, img_h:int, img_w:int):
        img = img.detach().cpu()
        img = ImageEmbedding.image_auto_unpadding(img, (img_h, img_w), self.phi_size)
        img = img[0]
        img = (img + 1) / 2
        img = img.clip(0, 1)

        return img

    def forward(self, x:torch.Tensor):
        # encode
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        (_, _, img_h, img_w) = x.size()
        x = ImageEmbedding.image_auto_padding(x, self.phi_size)
        x = x * 2 - 1

        # init
        y = self._3c_phi(x)
        x = self._3c_phiT(y)

        # forward
        x = self.enc(x)
        x_rgb = x[:, :3, ...]
        x_features = x[:, 3:, ...]

        x_t = x_rgb # >>>>>>
        for (x_coeff, noise_pred_coeff, timestep, grad_step, inner_res) in zip(self.x_coeff_list, self.noise_pred_coeff_list, self.timestep_list, self.grad_steps, self.res_list):
            x_t = self.grad_descent(x_t, y, grad_step.to(x_t.device))

            x = torch.cat([x_t, x_features], dim=1)
            x = inner_res(x)
            x_t = x[:, :3, ...]
            x_features = x[:, 3:, ...]

            x_t = self.reverse_diffusion_process_step(x_t, timestep, x_coeff, noise_pred_coeff)
        x = x_t # <<<<<<

        x = self.dec(x)

        # decode
        x = (x + 1) / 2
        x = ImageEmbedding.image_auto_unpadding(x, (img_h, img_w), self.phi_size)
        if not self.training:
            x = x.clip(0, 1)
        
        return x
    
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_true, x_pred):
        pred_loss = self._discrepancy_loss(x_true, x_pred)
        loss = pred_loss

        return loss
    
    def _discrepancy_loss(self, x_true, x_pred):
        loss = (x_pred - x_true).abs().mean()
        return loss
    