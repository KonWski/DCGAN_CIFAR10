from dataset import CIFAR10GAN
import torch
from models import DiscriminatorCIFAR10, GeneratorCIFAR10
from torch.optim import Adam
from torch.nn import BCELoss
import logging

def train_model(
        device, 
        n_epochs: int,
        batch_size: int,
        checkpoints_dir: str,
        download_datasets: bool,
        root_datasets_dir: str,
        class_name: str,
        latent_vector_length: int
    ):
    '''
    trains GAN model and saves its checkpoints to location
    given in checkpoints_dir.

    Discriminator's training
    -----------
    ascending gradient of: log(D(x)) + log(1 - D(G(z)))

    loss calculation split into two parts due to the form of BCEloss:
    l = weight * ( label * log(x) + (1 - label) * log(1 - x) )

    part_0 (for real images, label = 1):
    l_0 = 1 * log(D(x)) + (1 - 1) * log(1 - D(G(z)))
    l_0 = log(D(x))

    part_1 (for noise, label = 0)
    l_1 = 0 * log(D(x)) + (1 - 0) * log(1 - D(G(z)))
    l_1 = log(1 - D(G(z)))

    l_discriminator = l_0 + l_1 

    Generator's training
    ----------
    Goal is to maximize D(G(z)). For simplification k steps (described in Algorithm 1 in GAN paper)
    was set to 1 and noise samples were used from Discriminator training part.

    Parameters
    ----------
    device 
    n_epochs: int
        number of training epochs
    batch_size: int
        number of images inside single batch
    checkpoints_dir: str
        path to directory where checkpoints will be stored
    download_datasets: bool
        True -> download dataset from torchvision repo
    root_datasets_dir: str
        path to directory where dataset should be downloaded (download_datasets = True)
        or where dataset is already stored
    class_name: str
        one of ten classes in CIFAR10 dataset
    latent_vector_length: int
        length of random vector which will be transformed into an image by generator
    '''
    
    # datasets and dataloaders
    dataset = CIFAR10GAN(f'{root_datasets_dir}/train/', class_name, train=True, download=download_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # number of observations
    len_dataset = len(dataset)

    # models
    generator = GeneratorCIFAR10(latent_vector_length)
    discriminator = DiscriminatorCIFAR10()

    # optimizers
    optimizer_discriminator = Adam(discriminator.parameters(), lr=1e-5)
    optimizer_generator = Adam(generator.parameters(), lr=1e-5)
    
    # criterion
    criterion = BCELoss()

    for epoch in range(n_epochs):

        for id, batch in enumerate(loader, 0):
            
            # calculated parameters
            running_loss_discriminator = 0.0
            running_loss_generator = 0.0

            ##########################
            # Discriminator's training
            ##########################
            optimizer_discriminator.zero_grad()

            # inputs for discriminator and generator
            real_images = batch[0]
            noise = torch.rand(batch_size, latent_vector_length)

            # labels
            labels_real_images = torch.full((1, batch_size), 1)
            labels_fake_images = torch.full((1, batch_size), 0)

            # send tensors to device
            real_images = real_images.to(device)
            noise = noise.to(device)
            labels_real_images = labels_real_images.to(device)
            labels_fake_images = labels_fake_images.to(device)

            generated_images = generator(noise)
            classified_real_images = discriminator(real_images)

            # part_0
            print(f"classified_real_images shape: {classified_real_images.shape}")
            print(f"labels_real_images: {labels_real_images.shape}")
            loss_0 = criterion(classified_real_images, labels_real_images)
            loss_0.backward()

            # part_1, second use of backward sums all gradients
            loss_1 = criterion(generated_images, labels_fake_images)
            loss_1.backward()

            loss_discriminator = loss_0 + loss_1
            optimizer_discriminator.step()

            ##########################
            # Generator's training
            ##########################
            optimizer_generator.zero_grad()

            generated_images = generator(noise)
            classified_generated_images = discriminator(generated_images)
            loss_generator = criterion(classified_generated_images, labels_fake_images)

            loss_generator.backward()
            optimizer_generator.step()

            # iteration statistics
            running_loss_discriminator += loss_discriminator.item()
            running_loss_generator += loss_generator.item()

        # epoch statistics
        epoch_loss_discriminator = round(running_loss_discriminator / len_dataset, 2)
        epoch_loss_generator = round(running_loss_generator / len_dataset, 2)

        logging.info(f"Epoch: {epoch}, loss_disciminator: {epoch_loss_discriminator}, loss_generator: {epoch_loss_generator}")