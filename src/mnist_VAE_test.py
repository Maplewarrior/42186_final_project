import torch
import torch.nn as nn
from torchvision import datasets, transforms
from src.training.trainer import Trainer
from src.models.VAE import VAE, ContinuousBernoulliDecoder, GaussianEncoder, MultivariateGaussianDecoder
from src.models.vae_priors import StdGaussianPrior
from src.utils.misc import load_config
from torchvision.utils import save_image

if __name__ == '__main__':
    M = 32
    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2*M),
    )

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    ### Define VAE model
    prior = StdGaussianPrior(M)
    
    encoder = GaussianEncoder(H=28, W=28, D=M)
    encoder.encoder_net = encoder_net

    decoder = ContinuousBernoulliDecoder(H=28, W=28, D=M)
    decoder.decoder_net = new_decoder()

    model = VAE(encoder, decoder, prior)
    
    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]
        return torch.utils.data.TensorDataset(new_data, new_targets)
    
    num_train_data = 4096
    num_test_data = 16  # we keep this number low to only compute a few geodesics
    num_classes = 3
    train_tensors = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    
    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    trainer = Trainer(model, mnist_train_loader, 'configs/config.yaml')
    trainer.train()
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=64)
    
    save_image(samples, 'MNIST_samples.png')
