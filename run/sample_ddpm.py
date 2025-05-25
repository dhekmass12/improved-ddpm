import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet.UNet import UNet
from models.DDPM import DDPM
from models.DDIM import DDIM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt_minus_one = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt_minus_one, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt_minus_one, x0_pred = scheduler.sample_prev_timestep(xt_minus_one, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt_minus_one, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    diff_scheduler = diffusion_config["scheduler"]
    diff_model = diffusion_config["model"]
    num_timesteps=diffusion_config['num_timesteps']
    beta_start=diffusion_config['beta_start']
    beta_end=diffusion_config['beta_end']
    start=diffusion_config['s']
    end=diffusion_config['e']
    tau=diffusion_config['tau']
    s=diffusion_config['s']

    ddpm = DDPM(num_timesteps, beta_start, beta_end, start, end, tau, s)
    ddim = DDIM(num_timesteps, beta_start, beta_end, start, end, tau, s)
    
    # Create the noise scheduler
    if diff_model == "ddpm":
        if diff_scheduler == "linear":
            scheduler = ddpm.linear_scheduler
        elif diff_scheduler == "cosine":
            scheduler = ddpm.cosine_scheduler
        elif diff_scheduler == "sigmoid":
            scheduler = ddpm.sigmoid_scheduler
    elif diff_model == "ddim":
        if diff_scheduler == "linear":
            scheduler = ddim.linear_scheduler
        elif diff_scheduler == "cosine":
            scheduler = ddim.cosine_scheduler
        elif diff_scheduler == "sigmoid":
            scheduler = ddim.sigmoid_scheduler
    
    # Load model with checkpoint
    model = UNet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()

    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist3.yaml', type=str)
    args = parser.parse_args()
    infer(args)