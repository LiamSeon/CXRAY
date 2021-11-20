from dataloads import *
from generator import *
from loss import *
from discriminator import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

root_dir = '/home/ubuntu/CheXpert-v1.0'
save_path = '/home/ubuntu/xray_model/model_state_dict'
png_path = '/home/ubuntu/xray_model/png'
print(os.listdir(png_path))
epoch = 1000
bs = 1
device = 0

dataset = Xray_Dataset(root_dir, extension='jpg')
dataloader = DataLoader(dataset, batch_size = bs, shuffle = True)
generator = GeneratorRRDB(1, filters=32, num_res_blocks = 16).to(device)
import torch.optim as optim    # PSNR metrics.
pixel_criterion       = nn.L1Loss().to(device)      # Pixel loss.
adversarial_criterion = nn.BCELoss().to(device) 
p_optimizer = optim.Adam(generator.parameters(), lr = 2e-4)
p_scheduler = optim.lr_scheduler.StepLR(p_optimizer, step_size = 400, gamma = 0.5)
step = 0

for e in range(epoch):
    for lr, hr in dataloader:
        lr = lr.to(device)
        hr = hr.to(device)
        # Initialize the gradient of the generator model.
        generator.zero_grad()
        # Generate super-resolution images.
        sr = generator(lr)
        # Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
        pixel_loss = pixel_criterion(sr, hr)
        # Update the weights of the generator model.
        pixel_loss.backward()
        p_optimizer.step()
        if step <= 400:
            p_scheduler.step()
        step += 1

        if step % 1000 == 20:

            print('\n\n')
            print(e, 'lr')
            plt.figure(figsize = (40,16))
            plt.imshow(lr[0][0].detach().cpu(), cmap = 'gray')
            plt.savefig(os.path.join(png_path, 'lr.png'))
            plt.figure(figsize = (40,16))
            print(e, 'hr')
            plt.imshow(hr[0][0].detach().cpu(), cmap = 'gray')
            plt.savefig(os.path.join(png_path, 'hr.png'))
            print(e, 'sr')
            plt.figure(figsize = (40,16))
            plt.imshow(sr[0][0].detach().cpu(), cmap = 'gray')
            plt.savefig(os.path.join(png_path, 'sr.png'))

    torch.save(generator.state_dict(), os.path.join(save_path, 'epoch'+str(e)+'_first_stage_generator.pth'))
        
