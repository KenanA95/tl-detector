import numpy as np
from skimage.feature import hog, local_binary_pattern
from matplotlib import pyplot as plt
from ast import literal_eval


class HogDescriptor:
    def __init__(self, block_size, cell_size, orientations):
        self.block_size = block_size
        self.cell_size = cell_size
        self.orientations = orientations

    def compute(self, image, multichannel=True, visualize=False):
        return hog(image, self.orientations, pixels_per_cell=self.cell_size, cells_per_block=self.block_size,
                   multichannel=multichannel, visualize=visualize, block_norm='L2-Hys')

    def display(self, image):
        fd, hog_image = self.compute(image, visualize=True)
        plt.imshow(hog_image, cmap='gray')
        plt.show()

    @classmethod
    def from_config_file(cls, settings):
        return cls(literal_eval(settings['block_size']), literal_eval(settings['cell_size']),
                   int(settings['orientations']))

    def __repr__(self):
        return " Block Size: {0} \n Cell Size: {1} \n Orientations: {2}" \
               .format(self.block_size, self.cell_size, self.orientations)


class LBPDescriptor:
    def __init__(self, radius, points, method='default'):
        self.points = points
        self.radius = radius
        self.method = method

    def compute(self, image, visualize=False):
        lbp = local_binary_pattern(image, self.points, self.radius, self.method)

        # Compute and normalize the histogram
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.points + 3), range=(0, self.points + 2))
        hist = hist.astype('float') / hist.sum()

        if visualize:
            return hist, lbp

        return hist

    def display(self, image):
        hist, lbp = self.compute(image, visualize=True)
        plt.imshow(lbp, cmap='gray')
        plt.show()

    @classmethod
    def from_config_file(cls, settings):
        return cls(int(settings['points']), int(settings['radius']))

    def __repr__(self):
        return "LBP with {0} radius and {1} points using {2} method".format(self.radius, self.points, self.method)


class HaarDescriptor:
    def __init__(self):
        pass

    def compute(self):
        pass

    def display(self):
        pass

    @classmethod
    def from_config_file(cls, settings):
        return cls()

    def __repr__(self):
        pass
