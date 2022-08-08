import torchvision
import dnnlib
import numpy as np
import torch
import legacy
import os
import PIL.Image
import torch_utils.vgg_perceptual_loss as vgg_network
import datetime


class Encoder:
    def __init__(self, network, filename):
        self.listOfNetworks = [
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',       # dogs/cats
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',  # paintings
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl'        # faces
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

    # load and return network based on pickle and own device
    def load_network(self, network_pkl):
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            g = legacy.load_network_pkl(f)['G_ema'].to(self.device)  # type: ignore
        return g

    # generate and return latent based on a seed
    def generate_random_latent(self,seed):
        return torch.from_numpy(np.random.RandomState(seed).randn(1, self.network.z_dim)).to(self.device)

    # generate empty label
    def generate_label(self):
        return torch.zeros([1, self.network.c_dim], device=self.device)

    def synthesize(self):
        best_seed = 0
        best_loss = 20000

        for a in range(50):
            guess_latent = self.generate_random_latent(a)  # TODO: remove seed?
            img = self.network(guess_latent, self.label)
            loss = self.loss_fn(img)
            #print(a, loss)
            if loss < best_loss:
                best_seed = a
                best_loss = loss

        ws = self.network.mapping(z=self.generate_random_latent(best_seed), c=None)

        # save initial image
        # img = self.network(guess_latent, self.label)
        img = self.network.synthesis(ws=ws, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.outdir}/test_image_mapped.png')

        ws.requires_grad_(True)
        optimizer = torch.optim.Adam([ws], lr=0.001)

        for j in range(0):
            img = self.network.synthesis(ws=ws, noise_mode='const')

            # Loss function
            loss = self.loss_fn(img)  # loss
            if j % 20 == 0:
                print("Step: ", j)
                print(loss)

            # backprop
            loss.backward()  # backward
            optimizer.step()

            # save intermediate image
            if j % 200 == 0:
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.outdir}/progress_img_iter_' + str(j) + '.png')

        return img

    def loss_fn(self, img):
        img_clone = img.clone()
        img_clone = (img_clone.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img_clone[0].cpu().numpy(), 'RGB').save(f'{self.outdir}/mse_check_img.png')

        target_clone = self.target_image.clone()
        target_clone = (target_clone.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(target_clone[0].cpu().numpy(), 'RGB').save(f'{self.outdir}/mse_check_target.png')

        print(torch.nn.functional.mse_loss(img, self.target_image))

        return 0*self.vgg.forward(img, self.target_image) + 10 * torch.nn.functional.mse_loss(img, self.target_image)
        #return torch.nn.functional.mse_loss(img, self.target_image)

    def encode_single_image(self):
        os.makedirs(self.outdir, exist_ok=True)

        self.load_image()
        img = self.synthesize()

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.outdir}/'+str(datetime.datetime.now())+'_mapped_test.png')

    # set self.latent
    def load_image(self):
        # img = np.asarray(PIL.Image.open(self.indir+"/"+self.filename))
        # img = np.moveaxis(img, 2, 0)
        # img = np.array([img])/255

        # pic = PIL.Image.open(self.indir+"/"+self.filename)

        # img2_clone = (img2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # PIL.Image.fromarray(img2_clone[0].cpu().numpy(), 'RGB').save(f'{self.outdir}/loaded_image2.png')

        # self.target_image = torch.from_numpy(img2)

        #self.target_image = torchvision.transforms.ToTensor()(img).unsqueeze_(0).to(self.device)
        #self.target_image = torchvision.transforms.functional.pil_to_tensor(PIL.Image.open(self.indir+"/"+self.filename))

    # modify a single mapping matrix and synthesize the result
    def modify_single_map(self, styleblock, ws, modification):
        ws[0, styleblock] *= modification
        img = self.network.synthesis(ws=ws, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        return img


if __name__ == "__main__":
    encoder = Encoder(2, "normal_white_face.png")
    encoder.load_image()
    #encoder.encode_single_image()
    # encoder.TODO
