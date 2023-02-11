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
        # self.linear1 = Linear(self.latent_vector_length, 4096)
        # self.batchnorm0 = BatchNorm2d(1024)
        # self.convtranspose1 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2)
        # self.batchnorm1 = BatchNorm2d(512)
        # self.convtranspose2 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2)
        # self.batchnorm2 = BatchNorm2d(256)
        # self.convtranspose3 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2)
        # self.batchnorm3 = BatchNorm2d(128)
        # self.convtranspose4 = ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=1)
        # self.batchnorm4 = BatchNorm2d(3)


        # self.latent_vector_length = latent_vector_length
        # self.linear1 = Linear(self.latent_vector_length, 16384)
        # self.batchnorm0 = BatchNorm2d(1024)
        # self.convtranspose1 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=1) # out (512, 8, 8)
        # self.batchnorm1 = BatchNorm2d(512)
        # self.convtranspose2 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=1) # out (256, 12, 12)
        # self.batchnorm2 = BatchNorm2d(256)
        # self.convtranspose3 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=6, stride=2) # out (128, 28, 28)
        # self.batchnorm3 = BatchNorm2d(128)
        # self.convtranspose4 = ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=1)
        # self.batchnorm4 = BatchNorm2d(3)

        self.latent_vector_length = latent_vector_length
        self.linear1 = Linear(self.latent_vector_length, 16384)
        self.batchnorm0 = BatchNorm2d(1024)
        self.convtranspose1 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=1) # out (512, 8, 8)
        self.batchnorm1 = BatchNorm2d(512)

        self.convtranspose2 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=1) # out (256, 12, 12)
        self.batchnorm2 = BatchNorm2d(256)

        self.convtranspose3 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1) # out (256, 16, 16)
        self.batchnorm3 = BatchNorm2d(128)

        self.convtranspose4 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=1) # out (128, 20, 20)
        self.batchnorm4 = BatchNorm2d(64)

        self.convtranspose5 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1) # out (128, 24, 24)
        self.batchnorm5 = BatchNorm2d(32)

        self.convtranspose6 = ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=1) # out (128, 28, 28)
        self.batchnorm6 = BatchNorm2d(16)

        self.convtranspose7 = ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=5, stride=1) # out (128, 32, 32)


        # self.latent_vector_length = latent_vector_length
        # self.linear1 = Linear(self.latent_vector_length, 16384)
        # self.batchnorm0 = BatchNorm2d(1024)
        # self.convtranspose1 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2) # out (512, 11, 11)
        # self.batchnorm1 = BatchNorm2d(512)
        # self.convtranspose2 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2) # out (256, 25, 25)
        # self.batchnorm2 = BatchNorm2d(256)
        # self.convtranspose3 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1) # out (128, 29, 29)
        # self.batchnorm3 = BatchNorm2d(128)
        # self.convtranspose4 = ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=1)
        # self.batchnorm4 = BatchNorm2d(3)


        if inititialize_weights_xavier:
            self.apply(init_weights_xavier)


    def forward(self, x: Tensor):

        # print(f"x shape at begin: {x.shape}")
        x = self.linear1(x)

        # x = x.view(-1, 512, 2, 2)
        # x = self.batchnorm0(x)
        # x = self.convtranspose1(x) # (256, 4, 4)
        # x = relu(self.batchnorm1(x))
        # x = self.convtranspose2(x) # (128, 8, 8)
        # x = relu(self.batchnorm2(x))
        # x = self.convtranspose3(x) # (128, 16, 16)
        # x = relu(self.batchnorm3(x))
        # x = self.convtranspose4(x) # (3, 32, 32)
        # x = tanh(x)

        # x = x.view(-1, 1024, 2, 2)
        x = x.view(-1, 1024, 4, 4)

        x = self.batchnorm0(x)
        x = self.convtranspose1(x)
        x = relu(self.batchnorm1(x))
        x = self.convtranspose2(x)
        x = relu(self.batchnorm2(x))
        x = self.convtranspose3(x)
        x = relu(self.batchnorm3(x))
        x = self.convtranspose4(x)
        x = relu(self.batchnorm4(x))
        x = self.convtranspose5(x)
        x = relu(self.batchnorm5(x))
        x = self.convtranspose6(x)
        x = relu(self.batchnorm6(x))
        x = self.convtranspose7(x)
        x = tanh(x)

        print(f"x shape at end: {x.shape}")
        
        return x


class DiscriminatorCIFAR10(nn.Module):
    '''
    Classifies image as fake (created by generator) or real (sampled from original dataset)

    inititialize_weights_xavier: bool
        init weigts of layers using Xavier weight initialisation
    '''
    def __init__(self, inititialize_weights_xavier: bool = False):
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3) # output dim: (3, 30, 30)
        self.batchnorm1 = BatchNorm2d(6)
        self.conv2 = Conv2d(6, 12, 4, 2) # output dim: (12, 14, 14)
        self.batchnorm2 = BatchNorm2d(12)
        self.conv3 = Conv2d(12, 24, 4, 2) # output dim: (24, 6, 6)
        self.batchnorm3 = BatchNorm2d(24)
        self.conv4 = Conv2d(24, 48, 4, 2) # output dim: (48, 2, 2)
        self.batchnorm4 = BatchNorm2d(48)
        self.conv5 = Conv2d(48, 1, 2) # output dim: (1, 2, 1)
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
        # print(f"x shape before conv4: {x.shape}")
        x = self.conv4(x)
        x = leaky_relu(self.batchnorm4(x), 0.02)
        # print(f"x shape before conv5: {x.shape}")
        x = sigmoid(self.conv5(x))
        # print(f"x shape after conv5: {x.shape}")
        # x = self.flatten(x)
        # print(f"x shape after flatten: {x.shape}")

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