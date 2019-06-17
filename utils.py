# -*- coding: UTF-8 -*-

import numpy as np

def preprocessing_image(image, image_w, image_h):
    image_c = 3
    r_image = np.reshape(image, (image_h, image_w, image_c))
    
    return r_image