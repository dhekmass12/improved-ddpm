import torch
import math


class CosineNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self.s = 0.008
        self.alphas_cum_prod = torch.tensor([])
        self.betas = torch.tensor([])

        for t in range(self.num_timesteps):
            new_alpha_cum_prod = self.f(t) / self.f(0)
            new_beta = 1 - (new_alpha_cum_prod / (new_alpha_cum_prod - 1))

            self.alphas_cum_prod.cat((self.alphas, new_alpha_cum_prod))
            self.betas.cat((self.betas, new_beta))

        self.one_minus_betas = 1 - self.betas
        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_one_minus_betas = torch.sqrt(1 - self.betas)

    def f(self, t):
        nom = t/self.num_timesteps + self.s
        denom = 1+self.s

        return pow(math.cos(nom/denom * math.pi / 2), 2)

    
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

        sqrt_beta = self.sqrt_betas.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_beta = self.sqrt_one_minus_betas.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_beta = sqrt_one_minus_beta.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_beta = sqrt_beta.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_one_minus_beta.to(original.device) * original
                + sqrt_beta.to(original.device) * noise)

    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = ((xt - (self.sqrt_betas.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.one_minus_betas.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_betas.to(xt.device)[t])
        mean = mean / torch.sqrt(self.one_minus_betas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.one_minus_betas.to(xt.device)[t - 1]) / (1.0 - self.one_minus_betas.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0