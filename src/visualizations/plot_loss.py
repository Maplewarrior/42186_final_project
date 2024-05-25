import matplotlib.pyplot as plt

def plot_loss(losses, model_type: str):
    xvals = list(range(1, len(losses)+1))
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot(xvals, losses)
    ax.set_xlabel('step')
    ax.set_ylabel('neg. ELBO') if model_type == 'VAE' else ax.set_ylabel('MSE') 
    plt.savefig(f'figures/{model_type}_loss.png')
    
    
