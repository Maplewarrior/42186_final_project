import torch
import torch.nn as nn
from torchvision import datasets, transforms
from src.training.trainer import Trainer
from src.models.VAE import VAE, ContinuousBernoulliDecoder, GaussianEncoder, MultivariateGaussianDecoder
from src.models.vae_priors import StdGaussianPrior
from src.utils.misc import load_config
from torchvision.utils import save_image
from src.models.utils import Reshape

if __name__ == '__main__':
    M = 64
    H = 32
    W = 32
    encoder_net = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2*M),
    )
    ### Define VAE model
    def new_decoder():
        decoder_net_cnn = nn.Sequential(
                                nn.Linear(M, 1024),
                                Reshape((1024, 1, 1)),
                                nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=0),
                                nn.Softplus(),
                                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,stride=2, padding=1),
                                nn.Softplus(),
                                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,stride=2, padding=1),
                                nn.Softplus(),
                                nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4,stride=2, padding=1),
                                # nn.Softplus(),
                                # nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                                # nn.ReLU(),
                                # nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4,stride=2, padding=1),
                                nn.Sigmoid()
                            ) # (40-1) * s + k - 2*p
                                        
        
        return decoder_net_cnn
    
    prior = StdGaussianPrior(M)
    encoder = GaussianEncoder(H=28, W=28, D=M)
    encoder.encoder_net = encoder_net
    decoder = ContinuousBernoulliDecoder(H=28, W=28, D=M)
    decoder.decoder_net = new_decoder()
    model = VAE(encoder, decoder, prior)

    ### get cifar-10 dataset
    cifar_train = datasets.CIFAR10('data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=32)
    model.train()
    trainer = Trainer(model, cifar_train_loader, 'configs/config.yaml')
    trainer.train()
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=64)
    
    save_image(samples, 'CIFAR10_samples.png')