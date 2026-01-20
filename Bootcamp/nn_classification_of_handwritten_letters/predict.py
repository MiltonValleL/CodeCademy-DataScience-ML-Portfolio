
import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. MODEL ARCHITECTURE DEFINITION
# ==========================================
# 🏆 PROACTIVE: Self-Contained Architecture
# We explicitly redefine the architecture classes here to ensure 
# the script is completely standalone and requires no external dependencies
# beyond standard PyTorch/Numpy.

class ConvBlock(nn.Module):
    """
    A reusable block consisting of:
    Conv2d -> BatchNorm -> GELU -> MaxPool -> Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool=True, dropout_p=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class EMNISTNet(nn.Module):
    """
    Custom CNN architecture designed for EMNIST classification.
    Matches the architecture trained in the notebook exactly.
    """
    def __init__(self, num_classes=26):
        super(EMNISTNet, self).__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout_p=0.1),
            ConvBlock(32, 64, dropout_p=0.1),
            ConvBlock(64, 128, dropout_p=0.2)
        )

        # Classifier Head
        self.flatten_dim = 128 * 3 * 3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. MODEL LOADING & INFERENCE
# ==========================================

def load_model(model_path='final_model.pt', device='cpu'):
    """
    Load the trained model weights from disk.

    Args:
        model_path (str): Path to the .pt file.
        device (str): 'cpu' or 'cuda'.

    Returns:
        model (nn.Module): The model with loaded weights, set to eval mode.
    """
    try:
        model = EMNISTNet(num_classes=26)
        # map_location ensures it loads on CPU even if trained on GPU
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # 🏆 Critical: Set to eval to freeze BatchNorm/Dropout
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict(model, images, device='cpu'):
    """
    Make predictions on preprocessed images.

    Args:
        model: The loaded model.
        images: numpy array of shape (N, 28, 28) OR (28, 28).
                Values should be normalized floats [0, 1] or ints [0, 255].
        device: 'cpu' or 'cuda'.

    Returns:
        numpy array of predicted labels (0-25).
    """
    # 🏆 PROACTIVE: Robust Input Handling
    # Handle Numpy to Tensor conversion
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()

    # Handle single image input (28, 28) -> (1, 1, 28, 28)
    if images.dim() == 2:
        images = images.unsqueeze(0).unsqueeze(0)
    # Handle batch of images (N, 28, 28) -> (N, 1, 28, 28)
    elif images.dim() == 3:
        images = images.unsqueeze(1)

    # 🏆 PROACTIVE: Auto-Normalization check
    # If data seems to be 0-255, scale it to 0-1 automatically
    if images.max() > 1.0:
        images = images / 255.0

    # Standardize normalization (using dataset stats)
    # (x - mean) / std
    mean, std = 0.1722, 0.3309
    images = (images - mean) / std

    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        # Get index of max logit
        _, predicted = torch.max(outputs, 1)

    return predicted.cpu().numpy()

def label_to_letter(label):
    """Helper: Convert numeric label (0-25) to letter (A-Z)."""
    return chr(label + ord('A'))

# ==========================================
# 3. MAIN EXECUTION BLOCK (TEST)
# ==========================================
if __name__ == '__main__':
    print("🚀 Testing predict.py functionality...")

    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")

    # Load model
    model = load_model('final_model.pt', device=device)

    if model:
        # Create dummy data: 5 random images
        print("   Generating dummy test batch (5 images)...")
        test_images = np.random.rand(5, 28, 28).astype(np.float32)

        # Run prediction
        predictions = predict(model, test_images, device=device)

        print(f"   ✅ Prediction output shape: {predictions.shape}")
        print(f"   Sample Predictions (Indices): {predictions}")
        print(f"   Sample Predictions (Letters): {[label_to_letter(p) for p in predictions]}")
        print("\n🎉 Script is functioning correctly.")
