from torch import nn, Tensor, save, load
from torch.nn import Linear, Dropout, Conv2d, Flatten, Sequential, ReLU, Sigmoid, Softmax
from torch.nn.functional import softmax
import logging
from torch.nn.init import xavier_uniform

class GeneratorCIFAR10(nn.Module):
    '''
    Generates an image using given noise vector

    Attributes
    ----------
    latent_vector_length: int
        length of input noise vector
    init_randomly_weights: bool
        init weigts of layers using Xavier weight initialisation
    '''
    def __init__(self, latent_vector_length: int, init_randomly_weights: bool = False):

        super().__init__()
        self.latent_vector_length = latent_vector_length
        self.main = Sequential(
            Linear(self.latent_vector_length, 768),
            ReLU(inplace=True),
            Linear(768, 1536),
            Sigmoid(),
            Linear(1536, 2304),
            ReLU(inplace=True),
            Linear(2304, 3072),
            Sigmoid()
        )

        if init_randomly_weights:
            self.apply(init_weights_xavier)


    def forward(self, x: Tensor):

        x = self.main(x)
        x = x.view(-1, 3, 32, 32)

        return x


class DiscriminatorCIFAR10(nn.Module):
    '''
    Returns probability of image being sampled from original training set

    init_randomly_weights: bool
        init weigts of layers using Xavier weight initialisation

    Note:
        Original Discriminator used MaxOut activation function
    '''
    def __init__(self, init_randomly_weights: bool = False):

        super().__init__()
        self.main = Compose(
            Conv2d(3, 6, 3),
            ReLU(inplace=True),
            Dropout(p=0.2, inplace=True),
            Conv2d(6, 12, 6),
            ReLU(inplace=True),
            Flatten(),
            Linear(7500, 1000),
            ReLU(inplace=True),
            Linear(1000, 100),
            ReLU(inplace=True),
            Linear(100, 2),
            ReLU(inplace=True),
            Softmax()
        )
    
        if init_randomly_weights:
            self.apply(init_weights_xavier)

    def forward(self, x: Tensor):
        
        x = self.main(x)
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
                - loss from saved epoch
                
    '''
    checkpoint = load(checkpoint_path)
    latent_vector_length = checkpoint["latent_vector_length"]

    # initiate model
    model = GeneratorCIFAR10(latent_vector_length)

    # load parameters from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])

    # print loaded parameters
    logging.info(f"Loaded model from checkpoint: {checkpoint_path}")
    logging.info(f"Class name: {checkpoint['class_name']}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(f"Loss: {checkpoint['epoch_loss']}")
    logging.info(f"Save dttm: {checkpoint['save_dttm']}")
    logging.info(8*"-")

    return model, checkpoint