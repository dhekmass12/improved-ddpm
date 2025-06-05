from scheduler.ddpm.LinearNoiseScheduler import LinearNoiseScheduler
from scheduler.ddpm.CosineNoiseScheduler import CosineNoiseScheduler
from scheduler.ddpm.SigmoidNoiseScheduler import SigmoidNoiseScheduler
from scheduler.ddpm.SquareRootNoiseScheduler import SquareRootNoiseScheduler

class DDPM:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, start=-3, end=3, tau=1.1, s=0.008, p=2):
        self.linear_scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)
        self.cosine_scheduler = CosineNoiseScheduler(num_timesteps, s, p)
        self.sigmoid_scheduler = SigmoidNoiseScheduler(num_timesteps, start, end, tau)
        self.square_root_scheduler = SquareRootNoiseScheduler(num_timesteps, s)