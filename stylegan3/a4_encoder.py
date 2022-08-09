import torchvision
import dnnlib
import numpy as np
import torch
import legacy
import os
import PIL.Image
import torch_utils.vgg_perceptual_loss as vgg_network
import datetime
import torchvision.transforms.functional
import matplotlib.pyplot as plt


class Encoder:
    """
    Encode an Image into the latent space and save it locally.

    Attributes:
        network: A integer which indicates which network should be choosen from a hardcoded list.
        filename: A string which defines the filename of the locally saved image.
    """

    def __init__(self, network, filename):
        self.listOfNetworks = [
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',
            # dogs/cats
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',
            # paintings
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
            # faces 1024
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl'
            # faces 256
        ]
        self.device = torch.device('cuda')
        self.network = self.load_network(self.listOfNetworks[network])
        self.label = self.generate_label()
        self.indir = './_screenshots'
        self.filename = filename
        self.latent = None
        self.outdir = './out'  # output directory if image is getting rendered
        self.target_image = None  # target image to tensor
        self.vgg = vgg_network.VGGPerceptualLoss().to(self.device)

    def load_network(self, network_pkl):
        """
        Load and return a network.
        :param network_pkl: A pkl string to load the network from
        :return: stylegan3 network
        """
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            g = legacy.load_network_pkl(f)['G_ema'].to(self.device)  # type: ignore
        return g

    def generate_random_latent(self, seed):
        """
        Generate a random latent based on a seed.
        :param seed: An integer which serves as a seed for the random latent.
        :return: torch.Tensor
        """
        return torch.from_numpy(np.random.RandomState(seed).randn(1, self.network.z_dim)).to(self.device)

    def generate_label(self):
        """
        Generate an empty label.
        :return: torch.Zeros
        """
        return torch.zeros([1, self.network.c_dim], device=self.device)

    @staticmethod
    def plot_metrics(lrs, losses, distances):
        """
        Plot all Metrics with matplotlib.
        :param lrs: List of Learningrates.
        :param losses: List of losses.
        :param distances: List of distances between synthesized image and target image.
        """
        fig, ax = plt.subplots(3)
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel('Learning rate')
        ax[0].plot(lrs, "*")

        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('Loss')
        ax[1].plot(losses)

        ax[2].set_xlabel('Iterations')
        ax[2].set_ylabel('Change of latent vector')
        ax[2].plot(distances)
        plt.show()

    def load_image(self):
        # load image with PIL
        pil_image = PIL.Image.open(self.indir + "/" + self.filename)

        # transform to tensor with values between -1 and 1
        transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        img_tensor = transform(pil_image).unsqueeze_(0).to(self.device)
        img_tensor = img_tensor / 128 - 1

        self.target_image = img_tensor

    def save_tensor_as_image(self, tensor, imagename, suffix='.png', timestamp=False):
        """
        Save a pytorch tensor as an RGB-image into the global out-directory.
        :param tensor: A pytorch tensor which holds the image information.
        :param imagename: A string which serves as the imagename.
        :param suffix: A string which indicates the image suffig. e.g. ".png" or ".jpg".
        :param timestamp: A boolean which indicates if the image should have a timestamp as prefix. The timestamp
        has the following syntax: %H%M%S_.
        """
        img = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        if timestamp:
            current_time = datetime.datetime.now()
            current_time = current_time.strftime("%H%M%S")
            imagename = current_time + imagename

        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.outdir}/' + imagename + suffix)

    def search_best_seed(self, amount=50, lambda_vgg=1, lambda_mse=0, offset=50):
        """
        Search the best seed to start image synthesizing from.
        :param amount: An integer which indicates the amount of seeds to look threw.
        :param lambda_vgg: An integer which sets the weight of the VGG-Network to calculate the loss.
        :param lambda_mse: An integer which sets the weight of the MSE-Network to calculate the loss.
        :param offset: An integer to choose a starting point for the seeds.
        :return: Integer
        """
        best_seed = 0
        best_loss = 20000

        for a in range(amount):
            guess_latent = self.generate_random_latent(a + offset)
            img = self.network(guess_latent, self.label)
            loss = self.loss_fn(img, lambda_vgg=lambda_vgg, lambda_mse=lambda_mse)
            if loss < best_loss:
                best_seed = a + offset
                best_loss = loss

        print("Best Seed: ", best_seed)
        return best_seed

    def search_best_loss_weights(self, random_latent, vgg_weights=None, mse_weights=None, use_ws=False):
        """
        Searches the best weights for our loss function.
        :param random_latent: The base latent to search the weights from.
        :param vgg_weights: A list of VGG Weights. Default is [0, 5, 10]
        :param mse_weights: A list of MSE Weights. Default is [0, 10, 30].
        :param use_ws: Whether the latent should be synthesized in ws or in z.
        :return: best latent, best vgg weight, best mse weight
        """
        if vgg_weights is None:
            vgg_weights = [0, 5, 10]
        if mse_weights is None:
            mse_weights = [0, 10, 30]
        best_loss = 100000
        best_latent = None
        best_vgg = 0
        best_mse = 0

        for vgg_weight in vgg_weights:
            for mse_weight in mse_weights:
                # break out if both are zero
                if vgg_weight == 0 and mse_weight == 0:
                    break

                img, latent = self.encode(epoch_cnt=1, lambda_vgg=vgg_weight, lambda_mse=mse_weight,
                                          latent=random_latent, plot_metrics=False, use_ws=use_ws)

                # calculate loss only on mse
                loss = self.loss_fn(img, lambda_vgg=0, lambda_mse=1)

                if loss < best_loss:
                    best_loss = loss
                    best_latent = latent
                    best_vgg = vgg_weight
                    best_mse = mse_weight

        print("BestVGG: ", best_vgg, " BestMSE: ", best_mse)

        return best_latent, best_vgg, best_mse

    def encode(self, latent, epoch_cnt=5, steps_cnt=30, lambda_vgg=5, lambda_mse=30, plot_metrics=True,
               use_ws=False, optimizer_lr=0.05, scheduler_gamma=0.9):
        """
        Encode tries to encode an image into a latent vector.
        :param latent: The latent where to start synthesizing from.
        :param epoch_cnt: How many epoch should the optimizer cycle threw.
        :param steps_cnt: How many steps should the optimizer do in one epoch.
        :param lambda_vgg: The weight of the VGG Loss function.
        :param lambda_mse: The weight of the MSE Loss function.
        :param plot_metrics: Whether metrics should be printed.
        :param use_ws: Whether the latent should be synthesized in ws or in z.
        :param optimizer_lr: Learningrate of the optimizer.
        :param scheduler_gamma: Gamma of the scheduler.
        :return: img (Tensor), latent (Tensor)
        """
        # generate initial image based on ws or z
        if use_ws:
            ws = self.network.mapping(z=latent, c=None)
            img = self.network.synthesis(ws=ws, noise_mode='const')
            optimize_on = ws
        else:
            img = self.network(latent, self.label)
            optimize_on = latent

        # setup optimizer and scheduler
        optimize_on.requires_grad_(True)
        optimizer = torch.optim.Adam([optimize_on], lr=optimizer_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduler_gamma)

        # setup metric values
        lrs = []
        losses = []
        distances = []
        old_latent = None

        # save initial image
        if plot_metrics:
            self.save_tensor_as_image(img, "inital_image")

        # synthesize image
        for epoch in range(epoch_cnt):
            for step in range(steps_cnt):
                optimizer.zero_grad()

                if use_ws:
                    img = self.network.synthesis(ws=optimize_on, noise_mode='const')
                else:
                    img = self.network(optimize_on, self.label)

                # calculate loss
                loss = self.loss_fn(img, lambda_vgg=lambda_vgg, lambda_mse=lambda_mse)

                # show metrics
                if plot_metrics:
                    if step % 20 == 0:
                        print("Epoch: ", epoch, " Step: ", step)
                        print("Learning Rate: ", optimizer.param_groups[0]['lr'])
                        print(loss)
                    losses.append(loss.cpu().detach().numpy())
                    old_latent = optimize_on.cpu().detach().numpy()

                # backprop
                loss.backward()
                optimizer.step()

                # setup metrics
                if plot_metrics:
                    new_latent = optimize_on.cpu().detach().numpy()
                    distance = np.sqrt(np.sum(np.square(new_latent - old_latent)))
                    distances.append(distance)

            scheduler.step()

            # save intermediate image
            if plot_metrics:
                self.save_tensor_as_image(img, "progress_epoch_" + str(epoch))
                lrs.append(optimizer.param_groups[0]['lr'])

        if plot_metrics:
            self.plot_metrics(lrs, losses, distances)

        return img, latent

    def loss_fn(self, img, lambda_vgg=0, lambda_mse=1):
        """
        Calculate the loss of a given Image to our target image.
        The loss function is a combination of VGG and MSE. (lambda_vgg * VGG + lambda_mse * MSE).
        :param img: An Image (Tensor).
        :param lambda_vgg: Weight of the VGG Loss Function.
        :param lambda_mse: Weight of the MSE Loss Function.
        :return: The loss (Float)
        """
        return lambda_vgg * self.vgg.forward(img, self.target_image) + lambda_mse * torch.nn.functional.mse_loss(img, self.target_image)

    def encode_single_image(self, use_ws=False):
        """
        Encode a single image into a latent vector. In detail this functions serves as a multiplexer function which:
        - Checks if the outdir exists.
        - Loads the image into a tensor.
        - Searches the best seed.
        - Generates a random latent.
        - Searches for the best loss weights.
        - Encode an Image to a Latent.
        - Save the new interpretation as a .png file.
        :param use_ws: Whether the latent should be synthesized in ws or in z.
        """
        # check if outdir exists
        os.makedirs(self.outdir, exist_ok=True)

        # load image
        self.load_image()

        # search for best seed
        best_seed = self.search_best_seed(50, 1, 0)

        # generate latent based on best seed
        random_latent = self.generate_random_latent(best_seed)

        # search for best loss function weights
        best_latent, best_vgg, best_mse = self.search_best_loss_weights(random_latent, use_ws=use_ws)

        # synthesize image
        img, latent = self.encode(best_latent, epoch_cnt=5, steps_cnt=30, lambda_vgg=best_vgg, lambda_mse=best_mse,
                                  plot_metrics=True, use_ws=use_ws, optimizer_lr=0.05, scheduler_gamma=0.9)

        # save image
        self.save_tensor_as_image(img, "simple_synthesize", timestamp=True)
        # TODO: save latent in .txt/.yaml whatever


if __name__ == "__main__":
    # encode image. image path is ./_screenshots
    encoder = Encoder(2, "maxi_1024.jpg")
    encoder.encode_single_image(use_ws=False)
