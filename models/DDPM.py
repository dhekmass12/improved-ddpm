from scheduler.ddpm.LinearNoiseScheduler import LinearNoiseScheduler
from scheduler.ddpm.CosineNoiseScheduler import CosineNoiseScheduler

class DDPM:
    def __init__(self, num_timesteps=None, beta_start=None, beta_end=None):
        self.linear_scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)
        self.cosine_scheduler = CosineNoiseScheduler(num_timesteps)