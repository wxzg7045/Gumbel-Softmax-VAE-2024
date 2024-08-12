import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import zipfile
import random
import os

# Set the random seed and device
SEED = 1
torch.manual_seed(SEED)
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# Define constants
BATCH_SIZE = 128
Z_DIM = 300
SAVE_PATH = r"F:\rPPGdata_4\CelebA\best_model_CelebA_desktop2070_v2.pth"
DATA_DIR = r"F:\rPPGdata_4\CelebA\archive.zip"

# Define the CustomDataset class
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

# Define the GumbelSoftmaxVAE model (include only the essential parts for loading and inference)
class GumbelSoftmaxVAE(nn.Module):
    def __init__(self, tau_1=1.0, tau_2=1.0):
        super().__init__()
        self.tau_1 = nn.Parameter(torch.tensor(tau_1))
        self.tau_2 = nn.Parameter(torch.tensor(tau_2))

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        # Latent variables
        self.fc_logits_z1 = nn.Linear(256 * 8 * 8, Z_DIM)
        self.fc_logits_z2 = nn.Linear(Z_DIM, Z_DIM)

        # Decoder
        self.fc_dec1 = nn.Linear(Z_DIM, 256 * 8 * 8)
        self.fc_dec2 = nn.Linear(Z_DIM, 128 * 16 * 16)
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
        z = z.view(z.size(0), 256, 8, 8)
        z = F.relu(self.conv_trans1_1(z))
        z = F.relu(self.conv_trans2_1(z))
        z = F.relu(self.conv_trans3_1(z))
        z = torch.sigmoid(self.conv_trans4_1(z))
        return z

    def decode2(self, z):
        z = self.fc_dec2(z)
        z = z.view(z.size(0), 128, 16, 16)
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
        x_hat = (x_hat1 + x_hat2) / 2.0
        return x_hat

def plot_images(test_loader, model, device, rows=2, cols=20):
    # Get the first batch of images
    data_1, _ = next(iter(test_loader))
    data_1 = data_1.to(device)
    recon_batch_1 = model(data_1)

    # Get the second batch of images
    data_2, _ = next(iter(test_loader))
    data_2 = data_2.to(device)
    recon_batch_2 = model(data_2)

    # Convert to numpy arrays
    original_images_1 = data_1.cpu().numpy()
    reconstructed_images_1 = recon_batch_1.cpu().detach().numpy()
    original_images_2 = data_2.cpu().numpy()
    reconstructed_images_2 = recon_batch_2.cpu().detach().numpy()

    # Plotting
    fig, axes = plt.subplots(nrows=2 * rows, ncols=cols, figsize=(2 * cols, 4 * rows))
    for i in range(cols):
        # Display first set of original images in the first row
        axes[0, i].imshow(original_images_1[i].transpose(1, 2, 0))
        axes[0, i].axis('off')

        # Display first set of reconstructed images in the second row
        axes[1, i].imshow(reconstructed_images_1[i].transpose(1, 2, 0))
        axes[1, i].axis('off')

        # Display second set of original images in the third row
        axes[2, i].imshow(original_images_2[i].transpose(1, 2, 0))
        axes[2, i].axis('off')

        # Display second set of reconstructed images in the fourth row
        axes[3, i].imshow(reconstructed_images_2[i].transpose(1, 2, 0))
        axes[3, i].axis('off')

    plt.subplots_adjust(wspace=0.02, hspace=0.02)  # Adjust the spacing between images
    plt.show()

if __name__ == "__main__":
    # Define the transform for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Create a custom dataset
    custom_dataset = CustomDataset(DATA_DIR, transform=transform)
    total_size = len(custom_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Load the best model
    best_model = GumbelSoftmaxVAE().to(device)
    state_dict = torch.load(SAVE_PATH)
    best_model.load_state_dict(state_dict)
    best_model.tau_1 = nn.Parameter(torch.tensor(50000.0).to(device))
    best_model.tau_2 = nn.Parameter(torch.tensor(50000.0).to(device))
    best_model.eval()

    # Plot the original and reconstructed images
    plot_images(test_loader, best_model, device)
