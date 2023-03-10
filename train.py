from dataset import CIFAR10GAN
import torch
from models import DiscriminatorCIFAR10, GeneratorCIFAR10
from torch.optim import Adam
from torch.nn import BCELoss
import logging
from torchvision import transforms
from torchvision.utils import save_image

def train_model(
        device, 
        n_epochs: int,
        batch_size: int,
        ref_images_dir: str,
        download_datasets: bool,
        root_datasets_dir: str,
        class_name: str,
        latent_vector_length: int,
        init_generator_weights: bool,
        init_discriminator_weights: bool
    ):
    '''
    trains GAN model and saves referance images built for each epoch using the same 
    random noise. 

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
    ref_images_dir: str
        path to directory where ref images will be stored
    download_datasets: bool
        True -> download dataset from torchvision repo
    root_datasets_dir: str
        path to directory where dataset should be downloaded (download_datasets = True)
        or where dataset is already stored
    class_name: str
        one of ten classes in CIFAR10 dataset
    latent_vector_length: int
        length of random vector which will be transformed into an image by generator
    init_generator_weights: bool
        Init generator's weights using normal distribiution
    init_discriminator_weights: bool
        Init discriminator's weights using normal distribiution
    '''
    
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # datasets and dataloaders
    dataset = CIFAR10GAN(f'{root_datasets_dir}/train/', class_name, train=True, transform=transform_cifar, download=download_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # number of observations
    len_dataset = len(dataset)

    # models
    generator = GeneratorCIFAR10(latent_vector_length, init_generator_weights).to(device)
    discriminator = DiscriminatorCIFAR10(init_discriminator_weights).to(device)

    # optimizers
    optimizer_discriminator = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_generator = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # criterion
    criterion = BCELoss()

    # example reference img
    ref_noise = torch.randn(1, latent_vector_length, 1, 1)
    ref_noise = ref_noise.to(device)

    for epoch in range(n_epochs):
        
        # calculated parameters
        running_loss_discriminator = 0.0
        running_loss_generator = 0.0
        running_sum_real_proba = 0.0
        running_sum_fake_proba_D_train = 0.0
        running_sum_fake_proba_G_train = 0.0

        for id, batch in enumerate(loader, 0):

            ##########################
            # Discriminator's training
            ##########################

            # inputs for discriminator and generator
            real_images = batch[0]
            batch_size = real_images.shape[0]

            noise = torch.randn(batch_size, latent_vector_length, 1, 1)

            # labels
            labels_real_images = torch.ones(batch_size)
            noisy_labels_real_images = labels_real_images - (torch.rand(batch_size) * 0.1)
            labels_fake_images = torch.zeros(batch_size)
            noisy_labels_fake_images = labels_fake_images + (torch.rand(batch_size) * 0.1)

            # send tensors to device
            real_images = real_images.to(device)
            labels_real_images = labels_real_images.to(device)
            noisy_labels_real_images = noisy_labels_real_images.to(device)
            noisy_labels_fake_images = noisy_labels_fake_images.to(device)

            # generate images using random noise
            noise = noise.to(device)
            generated_images = generator(noise)

            # classify real and fake images
            classified_real_images = discriminator(real_images).view(-1)
            classified_generated_images = discriminator(generated_images).view(-1)

            # collect epoch statistics
            running_sum_real_proba += classified_real_images.sum().item()
            running_sum_fake_proba_D_train += classified_generated_images.sum().item()

            # calculate loss_0
            loss_0 = criterion(classified_real_images, noisy_labels_real_images)

            # calculate loss_1
            loss_1 = criterion(classified_generated_images, noisy_labels_fake_images)

            # update discriminator's weights
            loss_discriminator = (loss_0 + loss_1) / 2
            optimizer_discriminator.zero_grad()
            loss_discriminator.backward(retain_graph = True)
            optimizer_discriminator.step()

            running_loss_discriminator += loss_discriminator.item()

            ##########################
            # Generator's training
            ##########################

            # generated_images = generator(noise)
            classified_generated_images = discriminator(generated_images).view(-1)

            loss_generator = criterion(classified_generated_images, labels_real_images)
            optimizer_generator.zero_grad()
            loss_generator.backward()
            optimizer_generator.step()

            # collect epoch statistics
            running_loss_generator += loss_generator.item()
            running_sum_fake_proba_G_train += classified_generated_images.sum().item()

        # epoch statistics
        epoch_mean_proba_real = round(running_sum_real_proba / len_dataset, 4)
        epoch_mean_proba_fake_D_train = round(running_sum_fake_proba_D_train / len_dataset, 4)
        epoch_mean_proba_fake_G_train = round(running_sum_fake_proba_G_train / len_dataset, 4)

        logging.info(f"Epoch: {epoch}, loss_discriminator: {running_loss_discriminator}, loss_generator: {running_loss_generator}")
        logging.info(f"Epoch: {epoch}, epoch_mean_proba_real: {epoch_mean_proba_real}, " \
            f"epoch_mean_proba_fake_D_train: {epoch_mean_proba_fake_D_train}, " \
            f"epoch_mean_proba_fake_G_train: {epoch_mean_proba_fake_G_train}")

        # save reference image
        ref_img = generator(ref_noise)
        ref_img = ref_img[0]
        ref_img = (ref_img * 0.5) + 0.5
        save_image(ref_img, f"{ref_images_dir}/ref_img_{epoch}.png")