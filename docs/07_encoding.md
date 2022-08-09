# 07 - Stylegan3 Encoding
How to transfer Images to latent.

## Task
The task was to interpreter a given image into a latent vector.
On top on that we should research if it is better to optimize on z or ws.
Finally, we should encode images of ourselves and morph these together with a slider.

## Implementation
### Search best seed
A

### Search best loss weights
A

### Loss Function
A

### Encoding
#### Optimizer and Scheduler
ExponentialLR & Onecycle (show plot results maybe)

## Results
### Z vs WS
WS gute/sehr gute? Resultate bei höheren Step und Epochen - Count. TODO: Wie ist das bei eigenen Bildern?
Z gute Resultate bei niedrigen Steps und Epochen. Kaum verbesserung nach geringer Anzahl von Iterationen.
TODO: Insert Vergleiche WS bei wenigen Iteration und Z bei wenigen Iterationen. Auf Englisch übersetzen!
TODO: Show progress epoch of woman of WS vs Woman of Z?

### Plots

### Images

### Slider


## Sourcecode
```python
BarBaz
```

# Source
- How to Embed Images Into the StyleGAN Latent Space  
  https://openaccess.thecvf.com/content_ICCV_2019/papers/Abdal_Image2StyleGAN_How_to_Embed_Images_Into_the_StyleGAN_Latent_Space_ICCV_2019_paper.pdf
- VGG Perceptual Loss Implementation
  https://gist.github.com/alex-vasilchenko-md/dc5155f96f73fc4f67afffcb74f635e0
