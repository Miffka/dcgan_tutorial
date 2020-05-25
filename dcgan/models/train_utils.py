import os.path as osp

import torch
import torchvision.utils as vutils
import tqdm


def train_gan(
    netD,
    netG,
    dataloader,
    real_label,
    fake_label,
    criterion,
    optimizerD,
    optimizerG,
    fixed_noise,
    nz,
    device,
    num_epochs,
    writer=None,
    verbose=False,
    save_folder=None,
):
    # Lists to keep track of progress
    iters = 0

    print("Starting Training Loop...")
    netD.to(device)
    netG.to(device)
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in tqdm.tqdm(enumerate(dataloader, 0,), desc="Train", total=len(dataloader)):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            netG.train()
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                if writer is None or verbose:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                        % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
                    )
                if writer:
                    writer.add_scalar("Loss/Generator", errG.item(), global_step=iters)
                    writer.add_scalar("Loss/Discriminator", errD.item(), global_step=iters)
                    writer.add_scalar("Mean/Real_batch", D_x, global_step=iters)
                    writer.add_scalar("Mean/Fake_batch1", D_G_z1, global_step=iters)
                    writer.add_scalar("Mean/Fake_batch2", D_G_z2, global_step=iters)

            # Check how the generator is doing by saving G's output on fixed_noise
            if writer is not None:
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    netG.eval()
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    writer.add_image(
                        "Generated_images", vutils.make_grid(fake, padding=2, normalize=True), global_step=iters
                    )

            iters += 1

        if save_folder is not None:
            torch.save(
                {
                    "netG_state": netG.state_dict(),
                    "netD_state": netD.state_dict(),
                    "optimizerG_state": optimizerG.state_dict(),
                    "optimizerD_state": optimizerD.state_dict(),
                    "epoch": epoch,
                    "iteration": iters,
                },
                osp.join(save_folder, "model_last.pth"),
            )
