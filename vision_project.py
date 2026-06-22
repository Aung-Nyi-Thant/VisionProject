import torch
import urllib.request

# 🕵️‍♂️ Trick the server into thinking we are a standard web browser
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')]
urllib.request.install_opener(opener)

# --- Your original code continues below ---
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# ... rest of your code
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

print("📦 Loading and Downloading the MNIST Image Dataset...")

# 1. Image Preprocessing: Convert raw images into normalized PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Centers pixel values around 0.0
])

# 2. Automatically download the training images
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# A DataLoader automatically chops our 60,000 training images into batches of 64 at a time
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

print("🏁 Dataset ready! Building the Vision Brain...")

# ========================================================
# THE COMPUTER VISION ARCHITECTURE
# ========================================================
class VisionBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # 784 input pixels -> 128 hidden nodes -> 10 output classes (digits 0-9)
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10) # Outputting 10 raw scores (logits)

    def forward(self, x):
        # x starts as a batch of 2D images: [64, 1, 28, 28]
        # We flatten them into a 1D vector: [64, 784]
        x = x.view(x.size(0), -1) 
        
        x = self.layer1(x)
        x = self.relu(x)
        raw_logits = self.layer2(x)
        return raw_logits

# Initialize our system
my_vision_brain = VisionBrain()
loss_scorecard = nn.CrossEntropyLoss() # Perfect for mapping multiple choices (0-9)
teacher = optim.SGD(my_vision_brain.parameters(), lr=0.01)

# ========================================================
# THE TRAINING LOOP
# ========================================================
print("🚀 Training the Vision Model on handwritten digits (Running 1 Epoch)...")

for batch_idx, (images, labels) in enumerate(train_loader):
    # images shape: [64, 1, 28, 28] | labels shape: [64] (The target digit, e.g., 5)
    
    # 1. Get the AI's guesses
    guesses = my_vision_brain(images)
    
    # 2. Calculate the mistake score
    loss = loss_scorecard(guesses, labels)
    
    # 3. Nudge the weights
    teacher.zero_grad()
    loss.backward()
    teacher.step()
    
    if batch_idx % 300 == 0:
        print(f"       Batch {batch_idx:03d}/{len(train_loader)} | Mistake Score: {loss.item():.4f}")

print("\n🎉 Training complete! Your AI can now read handwritten numbers.")

# ========================================================
# THE TEST VAULT EVALUATION
# ========================================================
print("\n🔒 Evaluating on Unseen Validation Images...")

# Download the completely separate testing dataset (10,000 images)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

correct = 0
total = 0

# Turn off gradient tracking since we are just grading, not training
with torch.no_grad():
    for images, labels in test_loader:
        # Pass the unseen test images through our vision brain
        outputs = my_vision_brain(images)
        
        # Take the index of the highest score as the AI's final answer
        # torch.argmax returns a number from 0 to 9
        predictions = torch.argmax(outputs, dim=1)
        
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

final_accuracy = (correct / total) * 100
print("==============================================")
print(f"🎯 Vision Model Test Accuracy: {final_accuracy:.2f}%")
print("==============================================")