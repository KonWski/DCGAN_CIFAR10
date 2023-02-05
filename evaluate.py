from models import DiscriminatorCIFAR10, GeneratorCIFAR10
from torch.utils.data import DataLoader
import torch
import logging
from torch.nn import BCELoss

def evaluate_model(device, discriminator: DiscriminatorCIFAR10, generator: GeneratorCIFAR10, dataloader: DataLoader, len_dataset: int, epoch: int):
    '''
    checkes how well discriminator classifies data as real/fake -
    - calculates accuracy on real and generated data.
    
    Parameters
    ----------
    device 
    discriminator: DiscriminatorCIFAR10
        checkes whether sample comes form real dataset 
    generator: GeneratorCIFAR10
        creates fake input to discriminator
    dataloader
    len_dataset: int
        number of observations in dataset
    epoch: int
        number of training epoch
    '''

    running_corrects_real = 0
    running_corrects_fake = 0
    running_loss_generator = 0.0
    criterion = BCELoss()

    for id, batch in enumerate(dataloader, 0):
        
        real_images = batch[0]
        real_images_size = real_images.shape[0]
        noise = torch.randn(real_images_size, generator.latent_vector_length)

        # labels
        tensor_zeros = torch.full((real_images_size, 1), 0, dtype=torch.float)
        tensor_ones = torch.full((real_images_size, 1), 1, dtype=torch.float)
        labels_real_images = torch.cat((tensor_zeros, tensor_ones), dim=1).to(device)
        labels_fake_images = torch.cat((tensor_ones, tensor_zeros), dim=1).to(device)

        # send tensors to device
        real_images = real_images.to(device)
        labels_real_images = labels_real_images.to(device)
        labels_fake_images = labels_fake_images.to(device)

        # generate images using random noise
        noise = noise.to(device)
        generated_images = generator(noise)

        # classify real and fake images
        classified_real_images = discriminator(real_images)
        classified_generated_images = discriminator(generated_images)

        # loss for the
        loss_generator = criterion(classified_generated_images, labels_real_images)
        running_loss_generator += loss_generator.item()

        # correctly classified images
        running_corrects_real += torch.sum(torch.argmax(classified_real_images, 1) == torch.argmax(labels_real_images, 1)).item()
        running_corrects_fake += torch.sum(torch.argmax(classified_generated_images, 1) == torch.argmax(labels_fake_images, 1)).item()

        # epoch statistics
        epoch_acc_real = running_corrects_real / len_dataset
        epoch_acc_fake = running_corrects_fake / len_dataset

        return epoch_acc_real, epoch_acc_fake, running_loss_generator