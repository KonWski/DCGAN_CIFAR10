from dataset import CIFAR10GAN
import torch
from models import DiscriminatorCIFAR10, GeneratorCIFAR10, save_checkpoint
from torch.optim import Adam
from torch.nn import BCELoss
import logging
from datetime import datetime
from torchvision import transforms
from evaluate import evaluate_model

def train_model(
        device, 
        n_epochs: int,
        batch_size: int,
        checkpoints_dir: str,
        download_datasets: bool,
        root_datasets_dir: str,
        class_name: str,
        latent_vector_length: int,
        init_generator_weights_xavier: bool,
        init_discriminator_weights_xavier: bool
    ):
    '''
    trains GAN model and saves its checkpoints to location
    given in checkpoints_dir.

    Checkpoints of saved GAN should satisfy following conditions:
    - lowest epoch's generator loss
    - discriminator's accuracy on real dataset and fake images should be between 
    [0.45, 0.55]

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
    init_generator_weights_xavier: bool
        Init generator's weights using Xavier's initialization
    init_discriminator_weights_xavier: bool
        Init discriminator's weights using Xavier's initialization
    '''
    
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # datasets and dataloaders
    dataset = CIFAR10GAN(f'{root_datasets_dir}/train/', class_name, train=True, transform=transform_cifar, download=download_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # number of observations
    len_dataset = len(dataset)

    # models
    generator = GeneratorCIFAR10(latent_vector_length, init_generator_weights_xavier).to(device)
    discriminator = DiscriminatorCIFAR10(init_discriminator_weights_xavier).to(device)

    # optimizers
    optimizer_discriminator = Adam(discriminator.parameters(), lr=3e-4)
    optimizer_generator = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.99))
    
    # criterion
    criterion = BCELoss()

    # lowest losses for both models
    lowest_epoch_loss_generator = float("inf")

    for epoch in range(n_epochs):
        
        # calculated parameters
        running_loss_discriminator = 0.0
        running_loss_generator = 0.0
        running_corrects_real = 0
        running_corrects_fake = 0

        for id, batch in enumerate(loader, 0):

            ##########################
            # Discriminator's training
            ##########################

            # inputs for discriminator and generator
            real_images = batch[0]
            real_images_size = real_images.shape[0]

            # noise = torch.randn(real_images_size, latent_vector_length)
            noise = torch.randn(real_images_size, 512, 2, 2)

            # labels
            tensor_zeros = torch.full((real_images_size, 1), 0, dtype=torch.float)
            tensor_ones = torch.full((real_images_size, 1), 1, dtype=torch.float)
            labels_real_images = torch.cat((tensor_zeros, tensor_ones), dim=1).to(device)
            labels_fake_images = torch.cat((tensor_ones, tensor_zeros), dim=1).to(device)
            
            # zero grad
            optimizer_discriminator.zero_grad()

            # send tensors to device
            real_images = real_images.to(device)
            labels_real_images = labels_real_images.to(device)
            # print(f"labels_real_images: {labels_real_images}")
            labels_fake_images = labels_fake_images.to(device)
            # print(f"labels_fake_images: {labels_fake_images}")

            # generate images using random noise
            noise = noise.to(device)
            generated_images = generator(noise)

            # classify real and fake images
            classified_real_images = discriminator(real_images)
            # print(f"classified_real_images: {classified_real_images}")
            classified_generated_images = discriminator(generated_images)
            # print(f"classified_generated_images: {classified_generated_images}")

            # correctly classified images
            running_corrects_real += torch.sum(torch.argmax(classified_real_images, 1) == torch.argmax(labels_real_images, 1)).item()
            running_corrects_fake += torch.sum(torch.argmax(classified_generated_images, 1) == torch.argmax(labels_fake_images, 1)).item()

            # calculate loss_0
            loss_0 = criterion(classified_real_images, labels_real_images)

            # calculate loss_1
            loss_1 = criterion(classified_generated_images, labels_fake_images)

            # update discriminator's weights
            loss_discriminator = (loss_0 + loss_1) / 2
            loss_discriminator.backward(retain_graph = True)
            optimizer_discriminator.step()

            ##########################
            # Generator's training
            ##########################
            optimizer_generator.zero_grad()

            # generated_images = generator(noise)
            classified_generated_images = discriminator(generated_images)
            loss_generator = criterion(classified_generated_images, labels_real_images)
            loss_generator.backward()
            optimizer_generator.step()

            # iteration statistics
            running_loss_discriminator += loss_discriminator.item()
            running_loss_generator += loss_generator.item()

        # epoch statistics
        epoch_acc_real = running_corrects_real / len_dataset
        epoch_acc_fake = running_corrects_fake / len_dataset

        logging.info(f"Epoch: {epoch}, loss_discriminator: {running_loss_discriminator}, loss_generator: {running_loss_generator}")
        logging.info(f"Epoch: {epoch}, epoch_acc_real: {epoch_acc_real}, epoch_acc_fake: {epoch_acc_fake}")        

        # save generator checkpoint
        # if running_loss_generator < lowest_epoch_loss_generator and abs(0.5 - epoch_acc_real) <= 0.05 \
        #     and abs(0.5 - epoch_acc_fake) <= 0.05:
        
        if running_loss_generator < lowest_epoch_loss_generator:

            lowest_epoch_loss_generator = min(lowest_epoch_loss_generator, running_loss_generator)

            checkpoint = {
                "model_state_dict": generator.state_dict(),
                "latent_vector_length": latent_vector_length,
                "class_name": class_name,
                "epoch": epoch,
                "epoch_generator_loss": running_loss_generator,
                "save_dttm": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            checkpoint_path = f"{checkpoints_dir}/GeneratorCIFAR10"
            save_checkpoint(checkpoint, checkpoint_path)