import torch
import math
import numpy as np


class SigmoidNoiseScheduler:
    r"""
    Class for the Sigmoid noise scheduler that is used in DDIM.
    """
    def __init__(self, num_timesteps, s, e, tau):
        self.num_timesteps = num_timesteps
        self.s = s
        self.e = e
        self.tau = tau
        self.alpha_cum_prods = torch.tensor([])
        self.betas = torch.tensor([])

        for t in range(self.num_timesteps):            
            sig = self.sigmoid((t*(self.e-self.s)+self.s)/self.tau)
            sig_s = self.sigmoid(self.s/self.tau)
            sig_e = self.sigmoid(self.e/self.tau)
            
            alpha_t_cum_prod = (-sig + sig_e) / (sig_e - sig_s)
            alpha_t_cum_prod = np.clip(alpha_t_cum_prod, 1e-9, 0.999)
            
            if t == 0:
                beta = 1.0 - alpha_t_cum_prod
            else:
                beta = 1.0 - (alpha_t_cum_prod / self.alpha_cum_prods[-1].item())

            self.alpha_cum_prods = torch.cat((self.alpha_cum_prods, torch.tensor([alpha_t_cum_prod])))
            self.betas = torch.cat((self.betas, torch.tensor([beta])))
        
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        alpha_cum_prods = self.alpha_cum_prods.to(original.device)[t].reshape(batch_size)
        betas = self.betas.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            alpha_cum_prods = alpha_cum_prods.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            betas = betas.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (torch.sqrt(alpha_cum_prods.to(original.device)) * original
                + torch.sqrt(1 - alpha_cum_prods.to(original.device)) * noise)

    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        
        x0 = xt - (torch.sqrt(1 - self.alpha_cum_prods.to(xt.device)[t])  * noise_pred)
        x0 = x0 / torch.sqrt(self.alpha_cum_prods.to(xt.device)[t])
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = xt - torch.sqrt(1 - self.alpha_cum_prods.to(xt.device)[t]) * noise_pred * xt
        mean = mean / torch.sqrt(self.alpha_cum_prods.to(xt.device)[t])
        if t == 0:
            mean = mean
        else:
            mean = mean * torch.sqrt(self.alpha_cum_prods.to(xt.device)[t - 1])
            mean = mean + torch.sqrt(1 - self.alpha_cum_prods.to(xt.device)[t - 1]) * noise_pred * xt
        
        return mean, x0