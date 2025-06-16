import torch
import yaml
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist import MnistDataset
from torch.utils.data import DataLoader
from models.unet.UNet import UNet
from models.DDPM import DDPM
from models.DDIM import DDIM
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG_NAME = "T_1000/ddim/mnist_cosine"


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    diff_model = diffusion_config["model"]
    diff_scheduler = diffusion_config["scheduler"]
    num_timesteps=diffusion_config['num_timesteps']
    beta_start = None
    beta_end = None
    start=None
    end=None
    tau=None
    s=None
    p=None
    
    # Create the noise scheduler
    if diff_scheduler == "linear":
        beta_start = diffusion_config['beta_start']
        beta_end = diffusion_config['beta_end']
    elif diff_scheduler == "cosine":
        s=diffusion_config['s']
        p=diffusion_config['p']
    elif diff_scheduler == "sigmoid":
        start=diffusion_config['start']
        end=diffusion_config['end']
        tau=diffusion_config['tau']
    elif diff_scheduler == "square_root":
        s=diffusion_config['s']
        
    ddpm = DDPM(num_timesteps, beta_start, beta_end, start, end, tau, s, p)
    ddim = DDIM(num_timesteps, beta_start, beta_end, start, end, tau, s, p)
    
    if diff_model == "ddpm":
        if diff_scheduler == "linear":
            scheduler = ddpm.linear_scheduler
        elif diff_scheduler == "cosine":
            scheduler = ddpm.cosine_scheduler
        elif diff_scheduler == "sigmoid":
            scheduler = ddpm.sigmoid_scheduler
        elif diff_scheduler == "square_root":
            scheduler = ddpm.square_root_scheduler
    
    elif diff_model == "ddim":
        if diff_scheduler == "linear":
            scheduler = ddim.linear_scheduler
        elif diff_scheduler == "cosine":
            scheduler = ddim.cosine_scheduler
        elif diff_scheduler == "sigmoid":
            scheduler = ddim.sigmoid_scheduler
        elif diff_scheduler == "square_root":
            scheduler = ddim.square_root_scheduler
    
    # Create the dataset
    mnist = MnistDataset('train', im_path=dataset_config['im_path'])
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    
    # Instantiate the model
    model = UNet(model_config).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), map_location=device))
    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    with open(f"debug/{"-".join(CONFIG_NAME.split("/"))}.txt", "w") as f:
        f.write("")

    #########################
    #   RUN THE TRAINING    #
    #########################

    print('Running the training....')

    training_start_time = time.time()
    
    # Run training
    losses_per_epoch = []
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        loss_mean = np.mean(losses)
        losses_per_epoch.append(loss_mean)

        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            loss_mean,
        ))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name']))

    training_end_time = time.time()
    
    print('Done Training')

    #########################
    #   TRAINING END    #
    #########################

    training_time = training_end_time - training_start_time
    training_hours =  int(training_time//3600)
    training_minutes = int(training_time%3600 // 60)
    training_seconds = int(training_time%3600%60)
    training_time = str(training_hours) + ":" + str(training_minutes) + ":" + str(training_seconds)

    with open(f"debug/{"-".join(CONFIG_NAME.split("/"))}.txt", "w") as f:
        f.write(f"Training time: {training_time}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses_per_epoch, color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"MNIST Training with {CONFIG_NAME}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{"-".join(CONFIG_NAME.split("/"))}.png")
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default=f'config/{CONFIG_NAME}.yaml', type=str)
    args = parser.parse_args()
    train(args)