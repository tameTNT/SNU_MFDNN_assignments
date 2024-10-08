import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


"""
Step 0 : Define training configurations
"""

batch_size = 64
learning_rate = 5e-4
num_epochs = 2000
# reg_coeff = 500 only required for VAE 
device = "cuda:0" if torch.cuda.is_available() else "cpu"


"""
Step 1 : Define custom dataset 
"""

def make_swiss_roll(n_samples=2000, noise = 1.0, dimension = 2, a = 20, b = 5):
    """
    Generate 2D swiss roll dataset 
    """
    t = 2 * np.pi * np.sqrt(np.random.uniform(0.25,4,n_samples))
    
    X = 0.1 * t * np.cos(t)
    Y = 0.1 * t * np.sin(t)
    
    errors = 0.025 * np.random.multivariate_normal(np.zeros(2), np.eye(dimension), size = n_samples)
    X += errors[:, 0]
    Y += errors[:, 1]
    return np.stack((X, Y)).T


def show_data(data, title):
    """
    Plot the data distribution
    """
    sns.set(rc={'axes.facecolor': 'honeydew', 'figure.figsize': (5.0, 5.0)})
    plt.figure(figsize = (5, 5))
    plt.rc('text', usetex = False)
    plt.rc('font', family = 'serif')
    plt.rc('font', size = 10)
    
    g = sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, thresh=0.1, levels=1000, cmap="Greens")
    
    g.grid(False)
    plt.margins(0, 0)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.title(title)
    plt.show()


sns.set(rc={'axes.facecolor': 'honeydew', 'figure.figsize': (5.0, 5.0)})
plt.figure(figsize = (5, 5))
plt.rc('text', usetex = False)
plt.rc('font', family = 'serif')
plt.rc('font', size = 10)


data = make_swiss_roll()
g = sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, thresh=0.1, levels=1000, cmap="Greens")
g.grid(False)
plt.margins(0, 0)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.title('p_data')
plt.savefig('swiss_roll_true.png')
plt.show()

"""
Step 2 : Define custom dataset and dataloader. 
"""

class SwissRollDataset(Dataset) : 
    def __init__(self, data) : 
        super().__init__()
        self.data = torch.from_numpy(data)
    
    def __len__(self) : 
        return len(self.data)
    
    def __getitem__(self, idx) :
        return self.data[idx]

    
data = make_swiss_roll()
dataset = SwissRollDataset(data)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)


"""
Step 3 : Implement models
"""
class Generator(nn.Module) : 
    def __init__(self) :
        super(Generator, self).__init__()
        self.l1 = nn.Linear(1, 32)
        self.l2 = nn.Linear(32, 2)
        self.activation = nn.Tanh()
        
    
    def forward(self, z) :
        z = z.float().view(-1, 1)
        out1 = self.activation(self.l1(z))
        return self.l2(out1)



class Discriminator(nn.Module) : 
    def __init__(self) : 
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(2, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
        self.activation = nn.Tanh() 
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, x) : 
        out1 = self.activation(self.l1(x.float()))
        out2 = self.activation(self.l2(out1))
        return self.sigmoid(self.l3(out2))


"""
Step 4 : Train models
""" 
netG = Generator().to(device)
netD = Discriminator().to(device)

# Optimizers 
optimizer_G = torch.optim.Adam(netG.parameters(), lr = learning_rate)
optimizer_D = torch.optim.Adam(netD.parameters(), lr = learning_rate)

# training loop
loss_netD, loss_netG = [], []


for epoch in range(num_epochs) : 
    for batch_idx, x in enumerate(loader) : 
        x = x.to(device) 
        
        # Train Discriminator network 
        optimizer_D.zero_grad()
        z = torch.randn(x.shape[0], 1).to(device)
        loss_D = torch.mean(-torch.log(netD(x))-torch.log(1-netD(netG(z))))
        loss_D.backward()
        optimizer_D.step()
        loss_netD.append(-loss_D.item())
        
        # Train Generator network 
        optimizer_G.zero_grad()
        new_z =  torch.randn(x.shape[0], 1).to(device)
        loss_G = torch.mean(torch.log(1-netD(netG(new_z))))
        loss_G.backward()
        optimizer_G.step()
        loss_netG.append(loss_G.item())    
    
    
    # Visualize the intermediate result
    if (epoch + 1) % (num_epochs // 5) == 0:
        z = torch.randn(3000, 1).to(device)
        with torch.no_grad():
            samples = netG(z).cpu().detach()
        
        sns.set(rc={'axes.facecolor': 'honeydew', 'figure.figsize': (5.0, 5.0)})
        plt.figure(figsize = (5, 5))
        plt.rc('text', usetex = False)
        plt.rc('font', family = 'serif')
        plt.rc('font', size = 10)

        g = sns.kdeplot(x=samples[:, 0], y=samples[:, 1], fill=True, thresh=0.1, levels=1000, cmap="Greens")

        g.grid(False)
        plt.margins(0, 0)
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.title(f"Epoch : {epoch + 1}")
        plt.show()