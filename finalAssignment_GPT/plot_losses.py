import re
import matplotlib.pyplot as plt
import numpy as np

# Raw log text
log_text = """
step 0: train loss 2.6962, train perplexity 14.8226, val loss 2.6865, val perplexity 14.6798
Saved model checkpoint to checkpoints/15_33_17_12_2024/iters-0.pth
step 500: train loss 1.6934, train perplexity 5.4380, val loss 1.6183, val perplexity 5.0444
Saved model checkpoint to checkpoints/15_33_17_12_2024/iters-500.pth
step 1000: train loss 1.4376, train perplexity 4.2106, val loss 1.4017, val perplexity 4.0623
Saved model checkpoint to checkpoints/15_33_17_12_2024/iters-1000.pth
step 1500: train loss 1.2794, train perplexity 3.5946, val loss 1.2742, val perplexity 3.5757
Saved model checkpoint to checkpoints/15_33_17_12_2024/iters-1500.pth
step 2000: train loss 1.1884, train perplexity 3.2818, val loss 1.1942, val perplexity 3.3009
Saved model checkpoint to checkpoints/15_33_17_12_2024/iters-2000.pth
step 2500: train loss 1.1184, train perplexity 3.0600, val loss 1.1613, val perplexity 3.1942
Saved model checkpoint to checkpoints/15_33_17_12_2024/iters-2500.pth
step 3000: train loss 1.0571, train perplexity 2.8780, val loss 1.1478, val perplexity 3.1513
Saved model checkpoint to checkpoints/15_33_17_12_2024/iters-3000.pth
step 3500: train loss 1.0050, train perplexity 2.7319, val loss 1.1553, val perplexity 3.1748
Saved model checkpoint to checkpoints/15_33_17_12_2024/iters-3500.pth
step 4000: train loss 0.9382, train perplexity 2.5554, val loss 1.1697, val perplexity 3.2210
Saved model checkpoint to checkpoints/15_33_17_12_2024/iters-4000.pth
step 4500: train loss 0.8650, train perplexity 2.3751, val loss 1.2034, val perplexity 3.3316
"""

# Initialize lists to store the data
steps = []
train_losses = []
train_perplexities = []
val_losses = []
val_perplexities = []

# Parse the log text
for line in log_text.strip().split('\n'):
    if line.startswith('step'):
        # Extract values using regex
        step = int(re.search(r'step (\d+):', line).group(1))
        train_loss = float(re.search(r'train loss ([\d.]+)', line).group(1))
        train_perp = float(re.search(r'train perplexity ([\d.]+)', line).group(1))
        val_loss = float(re.search(r'val loss ([\d.]+)', line).group(1))
        val_perp = float(re.search(r'val perplexity ([\d.]+)', line).group(1))

        # Append to lists
        steps.append(step)
        train_losses.append(train_loss)
        train_perplexities.append(train_perp)
        val_losses.append(val_loss)
        val_perplexities.append(val_perp)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('V1.1 Training Metrics Over Time', fontsize=18, y=0.99)

# Plot losses
ax1.plot(steps, train_losses, 'b-', label='Training Loss', marker='o')
ax1.plot(steps, val_losses, 'r-', label='Validation Loss', marker='o')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Plot perplexities
ax2.plot(steps, train_perplexities, 'b-', label='Training Perplexity', marker='o')
ax2.plot(steps, val_perplexities, 'r-', label='Validation Perplexity', marker='o')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Perplexity')
ax2.set_title('Training and Validation Perplexity')
ax2.legend()
ax2.grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()