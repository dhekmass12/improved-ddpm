import torch

class Test:
    def __init__(self):
        self.num_timesteps = 1000
        self.s = 0.008
        self.p = 2
        f = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)

        self.alpha_cum_prods = torch.cos(((f / self.num_timesteps) + self.s) / (1+self.s) * torch.pi / 2) ** self.p
        self.alpha_cum_prods = self.alpha_cum_prods / self.alpha_cum_prods[0]
        self.alpha_cum_prods = torch.clip(self.alpha_cum_prods, 1e-9, 0.999)

        self.betas = 1 - (self.alpha_cum_prods[1:] / self.alpha_cum_prods[:-1])
        self.betas = torch.clip(self.betas, 1e-9, 0.999)
        self.alphas = 1. - self.betas

test = Test()
print(len(test.alpha_cum_prods))
print(len(test.betas))