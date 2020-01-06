from lime.wrappers.scikit_image import BaseWrapper
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel
from skimage.color import rgb2gray


class CustomSegmentationAlgorithm(BaseWrapper):
    """ https://github.com/marcotcr/lime/blob/master/lime/wrappers/scikit_image.py
        Modification to include watershed
    """

    def __init__(self, algo_type, **target_params):
        self.algo_type = algo_type
        if (self.algo_type == 'quickshift'):
            BaseWrapper.__init__(self, quickshift, **target_params)
            kwargs = self.filter_params(quickshift)
            self.set_params(**kwargs)
        elif (self.algo_type == 'felzenszwalb'):
            BaseWrapper.__init__(self, felzenszwalb, **target_params)
            kwargs = self.filter_params(felzenszwalb)
            self.set_params(**kwargs)
        elif (self.algo_type == 'slic'):
            BaseWrapper.__init__(self, slic, **target_params)
            kwargs = self.filter_params(slic)
            self.set_params(**kwargs)
        elif (self.algo_type == 'watershed'):
            BaseWrapper.__init__(self, watershed, **target_params)
            kwargs = self.filter_params(watershed)
            self.set_params(**kwargs)

    def __call__(self, *args):
        img = args[0] if self.algo_type != 'watershed' else sobel(rgb2gray(args[0]))
        result = self.target_fn(args[0], **self.target_params)
        return result
