"""Generate images using pretrained network pickle."""
import os

import torchvision

import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import plotly.graph_objects as go


# Linecarcombination takes two seeds and creates a Linearcombination of the latents which are generated by the seeds
class Linearcombination:
    # constructor
    def __init__(self, network_counter, seed1=1, seed2=2, generate_latents=True, latent1=None, latent2=None,
                 slider_size=10, use_ws=False, normalized=True):
        self.fig = go.Figure()
        self.listOfNetworks = [
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',      # dogs/cats
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
        ]
        print(self.listOfNetworks)
        self.device = torch.device('cuda')
        self.network = self.load_network(self.listOfNetworks[network_counter])
        if generate_latents:
            self.latent1 = self.generate_latent(seed1)
            self.latent2 = self.generate_latent(seed2)
        else:
            self.latent1 = latent1
            self.latent2 = latent2
        self.label = self.generate_label()
        self.slider_size = slider_size  # interpolation steps
        self.outdir = './out'           # output directory if image is getting rendered
        self.use_ws = use_ws
        self.normalized = normalized
        self.mean = torch.tensor([0.0216, -0.1726, -0.2571])
        self.std_dev = torch.tensor([0.5703, 0.5211, 0.5236])
        self.inverse_norm = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                                                                             std=[1 / self.std_dev[0],
                                                                                                  1 / self.std_dev[1],
                                                                                                  1 / self.std_dev[2]]),
                                                            torchvision.transforms.Normalize(
                                                                mean=[-self.mean[0], -self.mean[1], -self.mean[2]],
                                                                std=[1., 1., 1.])
                                                            ])

    # create slider and synthesize every picture needed for morph
    def create_slider(self):
        # add traces, one for each slider step
        for weight in np.arange(0, self.slider_size, 1):
            # synthesize image with weight
            img = self.synthesize(weight/(self.slider_size - 1))

            # add image to trace
            transformedImage = go.Image(z=img[0].cpu().numpy())
            self.fig.add_trace(transformedImage)

        # slider steps
        steps = []
        for i in range(self.slider_size):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(self.fig.data)},
                      {"title": "Slider switched to step: " + str(i)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        # create slider and add steps
        sliders = [dict(
            active=self.slider_size - 1,
            steps=steps
        )]

        # add slider to figure
        self.fig.update_layout(sliders=sliders)
        self.fig.show()

    # load and return network based on pickle and own device
    def load_network(self, network_pkl):
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            g = legacy.load_network_pkl(f)['G_ema'].to(self.device)  # type: ignore
        return g

    # generate and return latent based on a seed
    def generate_latent(self, seed):
        return torch.from_numpy(np.random.RandomState(seed).randn(1, self.network.z_dim)).to(self.device)

    # generate empty label
    def generate_label(self):
        return torch.zeros([1, self.network.c_dim], device=self.device)

    # generate linearcombination of two latents based on a given weight
    @staticmethod
    def calculate_linearcombination(weight, latent1, latent2):
        return (weight * latent1) + ((1 - weight) * latent2)

    # synthesize one image based on the linearcombination of two latents
    def synthesize(self, weight):
        latent = self.calculate_linearcombination(weight, self.latent1, self.latent2)

        if self.use_ws:
            # ws = self.network.mapping(z=latent, c=None)
            img = self.network.synthesis(ws=latent, noise_mode='const')
        else:
            img = self.network(latent, self.label)

        # denormalize if needed
        if self.normalized:
            img = self.inverse_norm(img)

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        return img

    # synthesize one image based on the linearcombination of two latents and save it locally
    def synthesize_single_image(self, weight, image_name):
        print('Generating image: ', image_name)
        os.makedirs(self.outdir, exist_ok=True)

        img = self.synthesize(weight)

        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.outdir}/'+image_name+'.png')


if __name__ == "__main__":
    linearCombination = Linearcombination(0, 1, 2)
    # linearCombination.synthesize_single_image("latent3", 0)
    linearCombination.create_slider()

