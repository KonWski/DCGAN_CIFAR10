from torchvision.datasets import CIFAR10

class CIFAR10GAN(CIFAR10):

    def __init__(self, 
            root: str, 
            class_name: str,
            train: bool = True, 
            download: bool = False,
        ) -> None:
        '''
        Dataset limiting original CIFAR10 to specified class_name

        Full path to dataset is defined by:
        {self.root_dir}/{self.state}/stanford_cars/cars_{self.state}

        Attributes
        ----------
        class_name: str
            limits records by given class name f.e. cat
        root_dir: str
            path to CIFAR10 dataset content
        state: str
            train or test
        '''
        super().__init__(root = root, train = train, download = download)
        self.class_name = class_name
        self.class_id = self.class_to_idx[self.class_name]
        self.data, self.target = self._filter_by_class_name()


    def _filter_by_class_name(self):
        '''
        returns data and targets limited to given class_name
        '''

        # find elements index for class name
        elements_indices = [index for index, id in enumerate(self.target) if id == self.class_id]

        # limit data and target
        data = [self.data[index] for index in elements_indices]
        target = [self.target[index] for index in elements_indices]
        
        return data, target