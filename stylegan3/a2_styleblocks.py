"""Generate images using pretrained network pickle."""
import os
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import plotly.graph_objects as go

# TODO: create gif based on multiple w's
# TODO: modify zwischenrepräsentation w
class Styleblocks:
    # constructor
    def __init__(self, network_counter, seed1, seed2):
        self.fig = go.Figure()
        self.listOfNetworks = [
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',      # dogs/cats
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl'  # TODO
        ]
        self.device = torch.device('cuda')
        self.network = self.load_network(self.listOfNetworks[network_counter])
        self.latent1 = self.generate_latent(seed1)
        self.latent2 = self.generate_latent(seed2)
        self.label = self.generate_label()
        self.slider_size = 10  # interpolation steps
        self.outdir = './out'  # output directory if image is getting rendered

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
        img = self.network(latent, self.label)
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
