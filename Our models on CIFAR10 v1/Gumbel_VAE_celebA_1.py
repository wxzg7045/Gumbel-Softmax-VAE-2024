import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import zipfile
import random

# Set the random seed and device
SEED = 1
torch.manual_seed(SEED)
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# Define constants
BATCH_SIZE = 128
Z_DIM = 300
NUM_EPOCHS = 800
LOG_INTERVAL = 50
SAVE_PATH = r"F:\rPPGdata_4\CelebA\best_model_CelebA_desktop2070_v2.pth"
DATA_DIR = r"F:\rPPGdata_4\CelebA\archive.zip"
LOSS_THRESHOLD = 0.01
NUM_MODELS = 11
EARLY_STOPPING_PATIENCE = 15  # Patience for early stopping

class CustomDataset(Dataset):
    def __init__(self, root_zip, transform=None):
        self.root_zip = root_zip
        self.transform = transform
        self.image_paths = self.extract_zip(root_zip)
        random.shuffle(self.image_paths)  # Shuffle the image paths

    def extract_zip(self, root_zip):
        # Extract the ZIP file
        extract_dir = r"F:\rPPGdata_4\CelebA\extracted_images"  # Main directory where the zip is extracted
        sub_dir = 'celeba_hq_256'  # Subdirectory containing the images
        full_extract_dir = os.path.join(extract_dir, sub_dir)  # Full path to the images
        with zipfile.ZipFile(root_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # List image files in the subdirectory
        image_paths = [os.path.join(full_extract_dir, filename)
                       for filename in os.listdir(full_extract_dir)
                       if os.path.isfile(os.path.join(full_extract_dir, filename))]
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

# Define the transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Define the GumbelSoftmaxVAE model
class GumbelSoftmaxVAE(nn.Module):
    def __init__(self, tau_1=1.0, tau_2=1.0):
        super().__init__()

        self.tau_1 = tau_1
        self.tau_2 = tau_2

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        # Two layers of latent variables
        self.fc_logits_z1 = nn.Linear(256 * 8 * 8, Z_DIM)  # Modify input size
        self.fc_logits_z2 = nn.Linear(Z_DIM, Z_DIM)

        # Decoder for each layer
        self.fc_dec1 = nn.Linear(Z_DIM, 256 * 8 * 8)  # Modify output size
        self.fc_dec2 = nn.Linear(Z_DIM, 128 * 16 * 16)  # Modify input size

        self.conv_trans1_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_trans2_1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_trans3_1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv_trans4_1 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.conv_trans1_2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_trans2_2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv_trans3_2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        logits_z1 = self.fc_logits_z1(x)
        logits_z2 = self.fc_logits_z2(logits_z1)

        return logits_z1, logits_z2

    def reparameterize(self, logits, tau):
        gumbel_noise = torch.rand_like(logits).to(device)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-8) + 1e-8)
        z = (logits + gumbel_noise) / tau
        z = F.softmax(z, dim=-1)
        return z

    def decode1(self, z):
        z = self.fc_dec1(z)
        z = z.view(z.size(0), 256, 8, 8)  # reshape to match the new input size
        z = F.relu(self.conv_trans1_1(z))
        z = F.relu(self.conv_trans2_1(z))
        z = F.relu(self.conv_trans3_1(z))
        z = torch.sigmoid(self.conv_trans4_1(z))
        return z

    def decode2(self, z):
        z = self.fc_dec2(z)
        z = z.view(z.size(0), 128, 16, 16)  # reshape to match the new input size
        z = F.relu(self.conv_trans1_2(z))
        z = F.relu(self.conv_trans2_2(z))
        z = torch.sigmoid(self.conv_trans3_2(z))
        return z

    def forward(self, x):
        logits_z1, logits_z2 = self.encode(x)
        z1 = self.reparameterize(logits_z1, self.tau_1)
        z2 = self.reparameterize(logits_z2, self.tau_2)

        x_hat1 = self.decode1(z1)
        x_hat2 = self.decode2(z2)

        # Average the output of the two decoders
        x_hat = (x_hat1 + x_hat2) / 2.0

        return x_hat, logits_z1, logits_z2, z1, z2

    def gumbel_softmax_loss_function(self, recon_x, x, logits_z1, logits_z2, z1, z2):
        # Binary cross entropy
        xent_loss = F.binary_cross_entropy(recon_x, x, reduction='none')
        xent_loss = torch.sum(xent_loss, dim=[1, 2, 3])

        # KL divergence for each layer
        p_z1 = F.softmax(logits_z1, dim=-1)
        p_z1 = torch.clamp(p_z1, torch.finfo(p_z1.dtype).eps, 1. - torch.finfo(p_z1.dtype).eps)  # to prevent log(0)
        kl_loss_z1 = torch.sum(p_z1 * torch.log(p_z1 * Z_DIM + torch.finfo(p_z1.dtype).eps), dim=-1)

        p_z2 = F.softmax(logits_z2, dim=-1)
        p_z2 = torch.clamp(p_z2, torch.finfo(p_z2.dtype).eps, 1. - torch.finfo(p_z2.dtype).eps)  # to prevent log(0)
        kl_loss_z2 = torch.sum(p_z2 * torch.log(p_z2 * Z_DIM + torch.finfo(p_z2.dtype).eps), dim=-1)

        # Total loss is the sum of the losses for each layer
        vae_loss = torch.mean(xent_loss + kl_loss_z1 + kl_loss_z2)
        return vae_loss

def plot_images(original_images, reconstructed_images, n=10):
    original_images = original_images.cpu().numpy().squeeze()
    reconstructed_images = reconstructed_images.cpu().detach().numpy().squeeze()

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_images[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def train_and_evaluate():
    # Create a custom dataset
    custom_dataset = CustomDataset(DATA_DIR, transform=transform)

    total_size = len(custom_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Initialize models and optimizers
    models = [GumbelSoftmaxVAE().to(device) for _ in range(NUM_MODELS)]
    optimizers = [optim.Adam(model.parameters(), lr=1e-3) for model in models]

    # Initialize temperature strategies
    initial_taus = [1.0 for _ in range(NUM_MODELS)]
    temp_strategies = ['fixed', 'linear_increase', 'linear_decrease', 'exp_increase', 'exp_decrease']
    temp_strategies_extended = ['fixed'] + temp_strategies * ((NUM_MODELS - 1) // len(temp_strategies)) + temp_strategies[:((NUM_MODELS - 1) % len(temp_strategies))]
    tau_min, tau_max = 0.001, 50000
    lin_increase, lin_decrease = 1.0, 0.005
    exp_increase, exp_decrease = 1.05, 0.999
    learning_rate = 0.01
    patience_threshold = 5
    patience_counter = [0 for _ in range(NUM_MODELS)]

    # Ensure temp_strategies_extended has correct length
    assert len(temp_strategies_extended) == NUM_MODELS, f"Expected {NUM_MODELS} strategies, got {len(temp_strategies_extended)}"

    best_tau_per_epoch_1 = []  # To record the best tau_1 for each epoch
    best_tau_per_epoch_2 = []  # To record the best tau_2 for each epoch

    best_global_loss = float('inf')  # Initialize the best global loss with infinity
    early_stop_counter = 0  # Initialize the early stopping counter

    for epoch in range(NUM_EPOCHS):
        losses = []
        best_reconstruction = None  # Keep track of the best reconstruction
        original_images = None

        for i in range(NUM_MODELS):
            model = models[i]
            temp_strategy = temp_strategies_extended[i]

            if temp_strategy == 'fixed':
                model.tau_1 = nn.Parameter(torch.tensor(float(initial_taus[i])), requires_grad=True)
                model.tau_2 = nn.Parameter(torch.tensor(float(initial_taus[i])), requires_grad=True)
            elif temp_strategy == 'linear_increase':
                model.tau_1 = nn.Parameter(torch.tensor(float(min(initial_taus[i] + epoch * lin_increase, tau_max))), requires_grad=True)
                model.tau_2 = nn.Parameter(torch.tensor(float(min(initial_taus[i] + epoch * lin_increase, tau_max))), requires_grad=True)
            elif temp_strategy == 'linear_decrease':
                model.tau_1 = nn.Parameter(torch.tensor(float(max(initial_taus[i] - epoch * lin_decrease, tau_min))), requires_grad=True)
                model.tau_2 = nn.Parameter(torch.tensor(float(max(initial_taus[i] - epoch * lin_decrease, tau_min))), requires_grad=True)
            elif temp_strategy == 'exp_increase':
                model.tau_1 = nn.Parameter(torch.tensor(float(min(initial_taus[i] * (exp_increase ** epoch), tau_max))), requires_grad=True)
                model.tau_2 = nn.Parameter(torch.tensor(float(min(initial_taus[i] * (exp_increase ** epoch), tau_max))), requires_grad=True)
            elif temp_strategy == 'exp_decrease':
                model.tau_1 = nn.Parameter(torch.tensor(float(max(initial_taus[i] * (exp_decrease ** epoch), tau_min))), requires_grad=True)
                model.tau_2 = nn.Parameter(torch.tensor(float(max(initial_taus[i] * (exp_decrease ** epoch), tau_min))), requires_grad=True)

            epoch_loss = 0.0

            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                optimizers[i].zero_grad()
                recon_batch, logits_z1, logits_z2, z1, z2 = model(data)
                loss = model.gumbel_softmax_loss_function(recon_batch, data, logits_z1, logits_z2, z1, z2)
                loss.backward()
                optimizers[i].step()
                epoch_loss += loss.item()

                if epoch_loss < LOSS_THRESHOLD and (best_reconstruction is None or loss.item() < min(losses)):
                    best_reconstruction = recon_batch
                    original_images = data

            epoch_loss /= len(train_loader)
            losses.append(epoch_loss)
            print(f'Train Epoch: {epoch} \tModel: {i} \tTraining Loss: {epoch_loss} \tTemperature Strategy: {temp_strategy} \ttau_1: {model.tau_1.item()} \ttau_2: {model.tau_2.item()}')

        best_model_idx = np.argmin(losses)
        best_loss = losses[best_model_idx]

        # Check if the current best loss is better than the global best loss
        if best_loss < best_global_loss:
            best_global_loss = best_loss
            best_model_state_dict = copy.deepcopy(models[best_model_idx].state_dict())
            torch.save(models[best_model_idx].state_dict(), SAVE_PATH)  # Save the best model
            print(f'Best model saved with index: {best_model_idx} and loss: {best_loss}')
            early_stop_counter = 0  # Reset the early stopping counter
        else:
            early_stop_counter += 1  # Increment the early stopping counter

        # Early stopping check
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

        for i in range(NUM_MODELS):
            if i != best_model_idx:
                if losses[i] > best_loss:
                    patience_counter[i] += 1
                    if patience_counter[i] >= patience_threshold:
                        delta_tau = learning_rate * (losses[i] - best_loss)  # positive item
                        models[i].tau_1 = nn.Parameter(torch.tensor(float(min(models[i].tau_1.item() + delta_tau, tau_max))), requires_grad=True)
                        models[i].tau_2 = nn.Parameter(torch.tensor(float(min(models[i].tau_2.item() + delta_tau, tau_max))), requires_grad=True)
                        patience_counter[i] = 0

        best_tau_per_epoch_1.append(models[best_model_idx].tau_1.item())
        best_tau_per_epoch_2.append(models[best_model_idx].tau_2.item())

        print(f'Epoch: {epoch} \tBest Model Index: {best_model_idx} \tBest Temperature Strategy: {temp_strategies_extended[best_model_idx]} \tBest tau_1: {models[best_model_idx].tau_1.item()} \tBest tau_2: {models[best_model_idx].tau_2.item()}')

        if best_reconstruction is not None and original_images is not None:
            plot_images(original_images.cpu(), best_reconstruction.cpu())

    # Output the best tau for each epoch
    for epoch, (tau_1, tau_2) in enumerate(zip(best_tau_per_epoch_1, best_tau_per_epoch_2)):
        print(f"Epoch: {epoch}, Best Tau 1: {tau_1}, Best Tau 2: {tau_2}")

    # Plot the best_tau_per_epoch
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(best_tau_per_epoch_1)), best_tau_per_epoch_1, label='Best Tau 1 per Epoch')
    plt.plot(range(len(best_tau_per_epoch_2)), best_tau_per_epoch_2, label='Best Tau 2 per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Tau values')
    plt.title('Best Tau Values per Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
