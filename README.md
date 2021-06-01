# pytorch-complex

Install it using pip: 

pip install pytorch-complex

Usage:
Similar to PyTorch.
For using the Complex features of this library, just change the regular torch imports with torchcomplex imports.
For example:
import torchcomplex.nn as nn  instead of import torch.nn as nn
Then, simply nn.Conv2d for both torch and torchcomplex, for 2D Convolution

## Credits

If you like this repository, please click on Star!

If you use this package or benift from the codes of this repo, please cite the following in your publications:

> [Soumick Chatterjee, Chompunuch Sarasaen, Alessandro Sciarra, Mario Breitkopf, Steffen Oeltze-Jafra, Andreas Nürnberger, Oliver Speck: Going beyond the image space: undersampled MRI reconstruction directly in the k-space using a complex valued residual neural network (ISMRM, May 2021)](https://www.researchgate.net/publication/349589092_Going_beyond_the_image_space_undersampled_MRI_reconstruction_directly_in_the_k-space_using_a_complex_valued_residual_neural_network)

BibTeX entry:

```bibtex
@inproceedings{mickISMRM21ksp,
      author = {Chatterjee, Soumick and Sarasaen, Chompunuch and Sciarra, Alessandro and Breitkopf, Mario and Oeltze-Jafra, Steffen and Nürnberger, Andreas and                     Speck, Oliver},
      year = {2021},
      month = {05},
      pages = {1757},
      title = {Going beyond the image space: undersampled MRI reconstruction directly in the k-space using a complex valued residual neural network},
      booktitle={2021 ISMRM \& SMRT Annual Meeting \& Exhibition}
}
```
Thank you so much for your support.
