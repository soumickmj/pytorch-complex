# pytorch-complex

Install it using pip: 

pip install pytorch-complex

Usage:
Similar to PyTorch.
For using the Complex features of this library, just change the regular torch imports with torchcomplex imports.
For example:
import torchcomplex.nn as nn  instead of import torch.nn as nn
Then, simply nn.Conv2d for both torch and torchcomplex, for 2D Convolution

If you use this package or benift from the codes of this repo, please cite the following in your publications:-
@inproceedings{mickISMRM21ksp,
      author = {Chatterjee, Soumick and Sarasaen, Chompunuch and Sciarra, Alessandro and Breitkopf, Mario and Oeltze-Jafra, Steffen and Nürnberger, Andreas and                     Speck, Oliver},
      year = {2021},
      month = {05},
      pages = {1757},
      title = {Going beyond the image space: undersampled MRI reconstruction directly in the k-space using a complex valued residual neural network},
      booktitle={2021 ISMRM \& SMRT Annual Meeting \& Exhibition}
}
Thank you so much for your support.
