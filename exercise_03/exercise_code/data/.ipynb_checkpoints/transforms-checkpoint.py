"""
Definition of image-specific transform classes
"""

# pylint: disable=too-few-public-methods

import numpy as np


class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, image):
        assert type(image) == np.ndarray, "The input image needs to be a numpy array! Make sure you dont send the string path."
        ret_image = None
        ret_image = self.min + ((image - self._data_min) / (self._data_max - self._data_min)) * (self.max - self.min)

        
        
        ########################################################################
        # TODO:                                                                #
        # Rescale the given images:                                            #
        #   - from (self._data_min, self._data_max)                            #
        #   - to (self.min, self.max)                                          #
        # Hint 1:                                                              #
        #       Google the following algorithm:                                #
        #       "convert-a-number-range-to-another-range-maintaining-ratio"    #
        # Hint 2:                                                              #
        #       Don't change the image in-place (directly in the memory),      # 
        #       but return a copy with ret_image                               #                                      
        ########################################################################
        

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return ret_image
    

def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    :param images: numpy array of shape NxHxWxC
        (for N images with C channels of spatial size HxW)
    :returns: per-channels mean and std; numpy array of shape (C,). 
    """
    mean, std = None, None
    ########################################################################
    # TODO:                                                                #
    # Calculate the per-channel mean and standard deviation of the images  #
    # Hint 1: You can use numpy to calculate the mean and standard         #
    # deviation.                                                           #
    #                                                                      #
    # Hint 2: Make sure that the shapes of the resulting arrays are (C,)   #
    # and not [1, C], [C, 1] or anything else. Use print(mean.shape) to    #
    # test yourself.                                                       #
    ########################################################################
     # Compute mean across the first three axes (N, H, W), keeping the channel dimension
    mean = np.mean(images, axis=(0, 1, 2))
    
    # Compute standard deviation across the first three axes, keeping the channel dimension
    std = np.std(images, axis=(0, 1, 2))
    
    # Check shapes to make sure they are (C,)
    assert mean.shape == (images.shape[-1],), f"Mean shape is {mean.shape}, should be (C,)"
    assert std.shape == (images.shape[-1],), f"Std shape is {std.shape}, should be (C,)"
    print(mean.shape)
    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return mean, std


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # normalize the given images:                                          #
        #   - substract the mean of dataset                                    #
        #   - divide by standard deviation                                     #
        ########################################################################
        if isinstance(self.mean, (int, float)):
            mean = self.mean
        else:
            mean = self.mean.reshape(1, 1, 1, -1)  # Reshape for broadcasting (1, 1, 1, C)

        if isinstance(self.std, (int, float)):
            std = self.std
        else:
            std = self.std.reshape(1, 1, 1, -1)  # Reshape for broadcasting (1, 1, 1, C)

        # Normalize images
        images = (images - mean) / std
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images


class IdentityTransform:
    """Transform class that does nothing"""
    def __call__(self, images):
        return images