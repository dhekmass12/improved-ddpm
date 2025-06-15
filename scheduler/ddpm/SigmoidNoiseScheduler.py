import torch
import math
import numpy as np


class SigmoidNoiseScheduler:
    r"""
    Class for the Sigmoid noise scheduler that is used in DDPM.
    """
    def __init__(self, num_timesteps, s, e, tau):
        self.num_timesteps = num_timesteps
        self.s = s
        self.e = e
        self.tau = tau
        t = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)

        sig = torch.sigmoid((t*(self.e-self.s)+self.s)/self.tau)
        sig_s = torch.sigmoid(torch.tensor(self.s/self.tau))
        sig_e = torch.sigmoid(torch.tensor(self.e/self.tau))

        self.alpha_cum_prods = (-sig + sig_e) / (sig_e - sig_s)
        self.alpha_cum_prods = torch.clip(self.alpha_cum_prods, 1e-9, 0.999)
        self.betas = 1 - (self.alpha_cum_prods[1:] / self.alpha_cum_prods[:-1])
        self.betas = torch.clip(self.betas, 1e-9, 0.999)
        self.alphas = 1. - self.betas
    
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
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            alpha_cum_prods = alpha_cum_prods.unsqueeze(-1)
        
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
        
        print(t.item())
        
        x0 = xt - (torch.sqrt(1 - self.alpha_cum_prods.to(xt.device)[t])  * noise_pred)
        x0 = x0 / torch.sqrt(self.alpha_cum_prods.to(xt.device)[t])
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = xt - (self.betas.to(xt.device)[t] * noise_pred / torch.sqrt(1 - self.alpha_cum_prods.to(xt.device)[t]))
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
            
        with open("debug/debug2.txt", "a") as f:
            f.write(f"T : {t}\n")
            f.write(f"beta : {self.betas[t].item()}\n")
            f.write(f"alpha : {self.alpha_cum_prods[t].item()/self.alpha_cum_prods[t-1].item()}\n")
            f.write(f"alpha cum prod : {self.alpha_cum_prods[t].item()}\n")
            f.write(f"xt : {xt}\n")
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prods.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prods.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas.to(xt.device)[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0