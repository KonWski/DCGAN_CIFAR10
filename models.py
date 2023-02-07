from torch import nn, Tensor, save, load
from torch.nn import Linear, Dropout, Conv2d, Flatten, ConvTranspose2d
from torch.nn.functional import relu, sigmoid, tanh, leaky_relu
import logging
from torch.nn.init import xavier_uniform

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
        self.latent_vector_length = latent_vector_length
        self.linear1 = Linear(self.latent_vector_length, 2048)
        self.convtranspose1 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.convtranspose2 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.convtranspose3 = ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.convtranspose4 = ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=2, stride=2)

        # self.latent_vector_length = latent_vector_length
        # self.linear1 = Linear(self.latent_vector_length, 768)
        # self.linear2 = Linear(768, 1536)
        # self.linear3 = Linear(1536, 2304)
        # self.linear4 = Linear(2304, 3072)

        if inititialize_weights_xavier:
            self.apply(init_weights_xavier)


    def forward(self, x: Tensor):

        print(f"x shape at begin: {x.shape}")
        x = leaky_relu(self.linear1(x))
        x = x.view(-1, 512, 2, 2)
        x = leaky_relu(self.convtranspose1(x)) # (256, 4, 4)
        x = leaky_relu(self.convtranspose2(x)) # (128, 8, 8)
        x = leaky_relu(self.convtranspose3(x)) # (128, 16, 16)
        x = tanh(self.convtranspose4(x)) # (3, 32, 32)
        x = x.view(-1, 3, 32, 32)
        print(f"x shape at end: {x.shape}")
        
        print(f"max x : {x.max()}")
        print(f"min x : {x.min()}")
        # x = relu(self.linear1(x))
        # x = relu(self.linear2(x))
        # x = relu(self.linear3(x))
        # x = tanh(self.linear4(x))
        # x = x.view(-1, 3, 32, 32)
        return x


class DiscriminatorCIFAR10(nn.Module):
    '''
    Classifies image as fake (created by generator) or real (sampled from original dataset)

    inititialize_weights_xavier: bool
        init weigts of layers using Xavier weight initialisation
    '''
    def __init__(self, inititialize_weights_xavier: bool = False):
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3)
        self.conv2 = Conv2d(6, 12, 6)
        self.flatten = Flatten()
        self.dropout = Dropout(p=0.4)
        self.linear1 = Linear(7500, 1000)
        self.linear2 = Linear(1000, 100)
        self.linear3 = Linear(100, 2)

        if inititialize_weights_xavier:
            self.apply(init_weights_xavier)

    def forward(self, x: Tensor):

        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = self.flatten(x)
        x = leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        x = sigmoid(x)
        return x



def init_weights_xavier(m):

    if isinstance(m, Linear) or isinstance(m, Conv2d):
        xavier_uniform(m.weight)


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