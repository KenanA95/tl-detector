from skimage.feature import hog
from matplotlib import pyplot as plt
from ast import literal_eval


class Descriptor:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def compute(self, **kwargs):
        pass

    def compute_all(self, **kwargs):
        pass

    def display(self, **kwargs):
        pass

    def __repr__(self):
        pass


class HogDescriptor(Descriptor):
    def __init__(self, block_size, cell_size, orientations):
        self.block_size = block_size
        self.cell_size = cell_size
        self.orientations = orientations
        Descriptor.__init__(self)

    def compute(self, image, multichannel=True, visualize=False):
        return hog(image, self.orientations, pixels_per_cell=self.cell_size, cells_per_block=self.block_size,
                   multichannel=multichannel, visualize=visualize, block_norm='L2-Hys')

    def compute_all(self, images, multichannel=True, visualize=False):
        return [self.compute(im, multichannel, visualize) for im in images]

    def display(self, image):
        fd, hog_image = self.compute(image, visualize=True)
        plt.imshow(hog_image, cmap='gray')
        plt.show()

    @classmethod
    def from_config_file(cls, config_settings):
        return cls(literal_eval(config_settings['block_size']), literal_eval(config_settings['cell_size']),
                   int(config_settings['orientations']))

    def __repr__(self):
        return " Block Size: {0} \n Cell Size: {1} \n Orientations: {2}" \
               .format(self.block_size, self.cell_size, self.orientations)


class LBPDescriptor(Descriptor):
    def __init__(self, points, radius, method='default'):
        self.points = points
        self.radius = radius
        self.method = method
        Descriptor.__init__(self)

    def compute(self):
        pass

    def compute_all(self, images, multichannel=True, visualize=False):
        pass

    def display(self):
        pass

    @classmethod
    def from_config_file(cls, config_settings):
        return cls(int(config_settings['points']), int(config_settings['radius']))

    def __repr__(self):
        pass


class HaarDescriptor(Descriptor):
    def __init__(self):
        Descriptor.__init__(self)

    def compute(self):
        pass

    def compute_all(self):
        pass

    def display(self):
        pass

    @classmethod
    def from_config_file(cls, config_settings):
        return cls()

    def __repr__(self):
        pass
