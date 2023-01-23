from torch import nn, Tensor
from torch.nn import Linear, Dropout, Conv2d, Softmax, Flatten
from torch.nn.functional import relu, sigmoid

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
        x = x.view(-1, 32, 32, 3)

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
        x = Softmax(x)

        return x