from torch import nn, Tensor, save, load
from torch.nn import Linear, Dropout, Conv2d, Flatten, ConvTranspose2d, BatchNorm2d, BatchNorm1d
from torch.nn.functional import relu, sigmoid, tanh, leaky_relu
import logging
from torch.nn.init import xavier_uniform, normal_

class GeneratorCIFAR10(nn.Module):
    '''
    Generates an image using given noise vector

    Attributes
    ----------
    latent_vector_length: int
        length of input noise vector
    inititialize_weights_xavier: bool
        init weigts of layers using Xavier weight initialisation
    '''
    def __init__(self, latent_vector_length: int, inititialize_weights_xavier: bool = False):

        super().__init__()

        # self.latent_vector_length = latent_vector_length
        # self.convtranspose1 = ConvTranspose2d(in_channels=latent_vector_length, out_channels=1024, kernel_size=5, stride=1, bias=False) # out (512, 5, 5)
        # self.batchnorm1 = BatchNorm2d(1024)
        # self.convtranspose2 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=1, bias=False) # out (512, 9, 9)
        # self.batchnorm2 = BatchNorm2d(512)
        # self.convtranspose3 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=1, bias=False) # out (256, 13, 13)
        # self.batchnorm3 = BatchNorm2d(256)
        # self.convtranspose4 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, bias=False) # out (128, 29, 29)
        # self.batchnorm4 = BatchNorm2d(128)
        # self.convtranspose5 = ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=1, bias=False) # out (3, 32, 32)
        # self.batchnorm5 = BatchNorm2d(3)

        self.latent_vector_length = latent_vector_length
        self.convtranspose1 = ConvTranspose2d(in_channels=latent_vector_length, out_channels=512, kernel_size=5, stride=2, bias=False) # out (512, 5, 5)
        self.batchnorm1 = BatchNorm2d(512)
        self.convtranspose2 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, bias=False) # out (512, 13, 13)
        self.batchnorm2 = BatchNorm2d(256)
        self.convtranspose3 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, bias=False) # out (256, 29, 29)
        self.batchnorm3 = BatchNorm2d(128)
        self.convtranspose4 = ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=1, bias=False) # out (128, 32, 32)


        if inititialize_weights_xavier:
            self.apply(init_weights_xavier)


    def forward(self, x: Tensor):

        # print(f"x shape at begin: {x.shape}")
        # x = self.convtranspose1(x)
        # x = relu(self.batchnorm1(x))
        # x = self.convtranspose2(x)
        # x = relu(self.batchnorm2(x))
        # x = self.convtranspose3(x)
        # x = relu(self.batchnorm3(x))
        # x = self.convtranspose4(x)
        # x = relu(self.batchnorm4(x))
        # x = self.convtranspose5(x)
        # x = tanh(x)

        x = self.convtranspose1(x)
        x = relu(self.batchnorm1(x))
        x = self.convtranspose2(x)
        x = relu(self.batchnorm2(x))
        x = self.convtranspose3(x)
        x = relu(self.batchnorm3(x))
        x = self.convtranspose4(x)
        x = tanh(x)

        # print(f"x shape at end: {x.shape}")
        
        return x


class DiscriminatorCIFAR10(nn.Module):
    '''
    Classifies image as fake (created by generator) or real (sampled from original dataset)

    inititialize_weights_xavier: bool
        init weigts of layers using Xavier weight initialisation
    '''
    def __init__(self, inititialize_weights_xavier: bool = False):
        super().__init__()

        # self.conv1 = Conv2d(3, 128, 4, 1, bias=False) # output dim: (3, 30, 30)
        # # self.batchnorm1 = BatchNorm2d(128)
        # self.conv2 = Conv2d(128, 256, 5, 2, bias=False) # output dim: (12, 14, 14)
        # self.batchnorm2 = BatchNorm2d(256)
        # self.conv3 = Conv2d(256, 512, 5, 1, bias=False) # output dim: (24, 6, 6)
        # self.batchnorm3 = BatchNorm2d(512)
        # self.conv4 = Conv2d(512, 1024, 5, 1, bias=False) # output dim: (48, 2, 2)
        # self.batchnorm4 = BatchNorm2d(1024)        
        # self.conv5 = Conv2d(1024, 1, 5, 1, bias=False) # output dim: (1, 2, 1)
        # self.flatten = Flatten()

        self.conv1 = Conv2d(3, 128, 4, 1, bias=False) # output dim: (3, 30, 30)
        self.conv2 = Conv2d(128, 256, 5, 2, bias=False) # output dim: (12, 14, 14)
        self.batchnorm2 = BatchNorm2d(256)
        self.conv3 = Conv2d(256, 512, 5, 2, bias=False) # output dim: (24, 6, 6)
        self.batchnorm3 = BatchNorm2d(512)
        self.conv4 = Conv2d(512, 1, 5, 2, bias=False) # output dim: (24, 6, 6)
        self.flatten = Flatten()

        if inititialize_weights_xavier:
            self.apply(init_weights_xavier)

    def forward(self, x: Tensor):
        
        # print(f"x shape before conv1: {x.shape}")
        x = leaky_relu(self.conv1(x), 0.02)
        # print(f"x shape before conv2: {x.shape}")
        x = self.conv2(x)
        x = leaky_relu(self.batchnorm2(x), 0.02)
        # print(f"x shape before conv3: {x.shape}")
        x = self.conv3(x)
        x = leaky_relu(self.batchnorm3(x), 0.02)
        # print(f"x shape before conv5: {x.shape}")
        x = sigmoid(self.conv4(x))
        # print(f"x shape after conv5: {x.shape}")
        # x = self.flatten(x)
        # print(f"x shape after sigmoid: {x.shape}")

        return x



def init_weights_xavier(m):

    if isinstance(m, Linear) or isinstance(m, Conv2d) or isinstance(m, BatchNorm2d) or isinstance(m, ConvTranspose2d):
        # xavier_uniform(m.weight)
        normal_(m.weight, 0.0, 0.02)


def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    '''
    saves checkpoint on given checkpoint_path
    '''
    save(checkpoint, checkpoint_path)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(8*"-")


def load_checkpoint(checkpoint_path: str):
    '''
    loads model checkpoint from given path

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint

    Notes
    -----
    checkpoint: dict
                parameters retrieved from training process i.e.:
                - model_state_dict
                - last finished number of epoch
                - save time
                - class_name
                - generator's loss from saved epoch (train)
    '''
    checkpoint = load(checkpoint_path)
    latent_vector_length = checkpoint["latent_vector_length"]

    # initiate model
    model = GeneratorCIFAR10(latent_vector_length, False)

    # load parameters from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])

    # print loaded parameters
    logging.info(f"Loaded model from checkpoint: {checkpoint_path}")
    logging.info(f"Class name: {checkpoint['class_name']}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(f"epoch_generator_loss: {checkpoint['epoch_generator_loss']}")
    logging.info(f"Save dttm: {checkpoint['save_dttm']}")
    logging.info(8*"-")

    return model, checkpoint