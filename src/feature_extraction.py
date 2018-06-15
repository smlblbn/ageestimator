#####################################################################################
#																					#
#	 This is an example script for extracting features of an image. For your own    #
#	 self portraits, you should get inspiration from this script. After extracting  #
#	 the feature vector, you can use it with your trained network.                  #
#																					#
#	 Note that, this requires torchvision, Pillow and NumPy packages.				#
#	 You are not forced to totally understand how the feature extractor works.      #
#	 You can just ignore the warnings given by the script.							#
#																					#
#####################################################################################

from PIL import Image
from img_to_vec import Img2Vec
import numpy as np

if __name__ == '__main__':
    fe = Img2Vec(cuda=True) # change this if you cannot use Cuda version of the PyTorch.

    img = Image.open('../ismail.jpg')
    img = img.resize((224, 224))
    feats = fe.get_vec(img).reshape(1, -1)

    with open('../ismail.npy', 'wb') as file:
        np.save(file, feats)

    img = Image.open('../ismail_sunglasses.jpg')
    img = img.resize((224, 224))
    feats = fe.get_vec(img).reshape(1, -1)

    with open('../ismail_sunglasses.npy', 'wb') as file:
        np.save(file, feats)