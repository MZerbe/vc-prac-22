# 07 - Stylegan3 Encoding
How to transfer Images to latent.

## Task
Foo

## Research
### Latent Space Embedding
#### Learn Encoder
- map a given image to the latent space e.g. Variational Auto-Encoder#
- fast solution of image, has problems generalizing beyond the training dataset

#### Random initial latent
- select random latent code and optimize it using gradient descent
- general and stable version of encoding
- Perceptual Loss and Style Transer:
  - TODO
  -

## Implementation
Bar


## Results
Baz

## Sourcecode
```python
BarBaz
```

# Source
- How to Embed Images Into the StyleGAN Latent Space  
  https://openaccess.thecvf.com/content_ICCV_2019/papers/Abdal_Image2StyleGAN_How_to_Embed_Images_Into_the_StyleGAN_Latent_Space_ICCV_2019_paper.pdf
- Pytorch Stylegan Encoder (200Stars)
  https://github.com/jacobhallberg/pytorch_stylegan_encoder
- Puzer Stylegan Encoder (1000 Stars)
  https://github.com/Puzer/stylegan-encoder
- StyleGAN3 Port of Puzer Encoder (700 Stars)
  https://github.com/pbaylies/stylegan-encoder
