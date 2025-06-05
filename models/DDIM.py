from scheduler.ddim.LinearNoiseScheduler import LinearNoiseScheduler
from scheduler.ddim.CosineNoiseScheduler import CosineNoiseScheduler
from scheduler.ddim.SigmoidNoiseScheduler import SigmoidNoiseScheduler

class DDIM:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, start=-3, end=3, tau=1.1, s=0.008):
        self.linear_scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)
        self.cosine_scheduler = CosineNoiseScheduler(num_timesteps, s)
        self.sigmoid_scheduler = SigmoidNoiseScheduler(num_timesteps, start, end, tau)