from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose

class CIFAR10GAN(CIFAR10):

    def __init__(self, 
            root: str, 
            class_name: str,
            train: bool = True,
            transform: Compose = Compose([ToTensor()]),
            download: bool = False,
        ) -> None:
        '''
        Dataset limiting original CIFAR10 to specified class_name

        Attributes
        ----------
        class_name: str
            limits records by given class name f.e. cat
        root_dir: str
            path to CIFAR10 dataset content
        train: bool = True
            select train or test part of CIFAR10
        transform: Compose
            set of transformation performed on images
        download: bool = False
            download dataset from torchvision or used existing data located
            in root_dir folder
        '''
        super().__init__(root = root, train = train, transform = transform, download = download)
        self.class_name = class_name
        self.class_id = self.class_to_idx[self.class_name]
        self.data, self.targets = self._filter_by_class_name()


    def _filter_by_class_name(self):
        '''
        returns data and targets limited to given class_name
        '''

        # find elements index for class name
        elements_indices = [index for index, id in enumerate(self.targets) if id == self.class_id]

        # limit data and target
        data = [self.data[index] for index in elements_indices]
        targets = [self.targets[index] for index in elements_indices]
        
        return data, targets