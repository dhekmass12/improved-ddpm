from scheduler.ddim.LinearNoiseScheduler import LinearNoiseScheduler
from scheduler.ddim.CosineNoiseScheduler import CosineNoiseScheduler
from scheduler.ddim.SigmoidNoiseScheduler import SigmoidNoiseScheduler

class DDIM:
    def __init__(self, num_timesteps=None, beta_start=None, beta_end=None, start=-3, end=3, tau=1.1, s=0.008):
        self.linear_scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)
        self.cosine_scheduler = CosineNoiseScheduler(num_timesteps, s)
        self.sigmoid_scheduler = SigmoidNoiseScheduler(num_timesteps, start, end, tau)