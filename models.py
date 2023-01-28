from torch import nn, Tensor, save, load
from torch.nn import Linear, Dropout, Conv2d, Flatten
from torch.nn.functional import relu, sigmoid, softmax
import logging

class GeneratorCIFAR10(nn.Module):
    '''
    Generates an image using given noise vector

    Attributes
    ----------
    latent_vector_length: int
        length of input noise vector
    '''
    def __init__(self, latent_vector_length: int):

        super().__init__()
        self.latent_vector_length = latent_vector_length
        self.linear1 = Linear(self.latent_vector_length, 768)
        self.linear2 = Linear(768, 1536)
        self.linear3 = Linear(1536, 2304)
        self.linear4 = Linear(2304, 3072)


    def forward(self, x: Tensor):

        x = relu(self.linear1(x))
        x = sigmoid(self.linear2(x))
        x = relu(self.linear3(x))
        x = sigmoid(self.linear4(x))
        x = x.view(-1, 3, 32, 32)

        return x


class DiscriminatorCIFAR10(nn.Module):
    '''
    Returns probability of image being sampled from original training set

    Note:
    Original Discriminator used MaxOut activation function
    '''
    def __init__(self):

        super().__init__()
        self.conv1 = Conv2d(3, 6, 3)
        self.dropout = Dropout(p=0.2)
        self.conv2 = Conv2d(6, 12, 6)
        self.flatten = Flatten()
        self.linear1 = Linear(7500, 1000)
        self.linear2 = Linear(1000, 100)
        self.linear3 = Linear(100, 2)        

    def forward(self, x: Tensor):
        
        x = relu(self.conv1(x))
        x = self.dropout(x)
        x = relu(self.conv2(x))
        x = self.flatten(x)
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        x = softmax(x)

        return x


def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    '''
    saves checkpoint on given checkpoint_path
    '''
    save(checkpoint, checkpoint_path)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
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