from scheduler.ddim.LinearNoiseScheduler import LinearNoiseScheduler
from scheduler.ddim.CosineNoiseScheduler import CosineNoiseScheduler
from scheduler.ddim.SigmoidNoiseScheduler import SigmoidNoiseScheduler
from scheduler.ddim.SquareRootNoiseScheduler import SquareRootNoiseScheduler

class DDIM:
    def __init__(self, num_timesteps, beta_start, beta_end, start, end, tau, s, p):
        num_timesteps=1000 if num_timesteps is None else num_timesteps
        beta_start=0.0001 if beta_start is None else beta_start
        beta_end=0.02 if beta_end is None else beta_end
        start=-3 if start is None else start
        end=3 if end is None else end
        tau=1.1 if tau is None else tau
        s=0.008 if s is None else s
        p=2 if p is None else p
        
        self.linear_scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)
        self.cosine_scheduler = CosineNoiseScheduler(num_timesteps, s, p)
        self.sigmoid_scheduler = SigmoidNoiseScheduler(num_timesteps, start, end, tau)
        self.square_root_scheduler = SquareRootNoiseScheduler(num_timesteps, s)