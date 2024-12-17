import re
import matplotlib.pyplot as plt
import numpy as np

# Raw log text
log_text = """
step 0: train loss 2.6962, train perplexity 14.8226, val loss 2.6865, val perplexity 14.6798
Saved model checkpoint to checkpoints/model_iter_0.pt
step 500: train loss 1.6796, train perplexity 5.3636, val loss 1.6102, val perplexity 5.0038
Saved model checkpoint to checkpoints/model_iter_500.pt
step 1000: train loss 1.3363, train perplexity 3.8049, val loss 1.3167, val perplexity 3.7309
Saved model checkpoint to checkpoints/model_iter_1000.pt
step 1500: train loss 1.1942, train perplexity 3.3009, val loss 1.2047, val perplexity 3.3358
Saved model checkpoint to checkpoints/model_iter_1500.pt
step 2000: train loss 1.1087, train perplexity 3.0304, val loss 1.1500, val perplexity 3.1583
Saved model checkpoint to checkpoints/model_iter_2000.pt
step 2500: train loss 1.0230, train perplexity 2.7816, val loss 1.1508, val perplexity 3.1608
Saved model checkpoint to checkpoints/model_iter_2500.pt
step 3000: train loss 0.9260, train perplexity 2.5245, val loss 1.1711, val perplexity 3.2255
Saved model checkpoint to checkpoints/model_iter_3000.pt
step 3500: train loss 0.8073, train perplexity 2.2418, val loss 1.2351, val perplexity 3.4386
Saved model checkpoint to checkpoints/model_iter_3500.pt
step 4000: train loss 0.6774, train perplexity 1.9688, val loss 1.3119, val perplexity 3.7133
Saved model checkpoint to checkpoints/model_iter_4000.pt
step 4500: train loss 0.5621, train perplexity 1.7543, val loss 1.4189, val perplexity 4.1324
Saved model checkpoint to checkpoints/model_iter_4500.pt
step 4999: train loss 0.4624, train perplexity 1.5879, val loss 1.5578, val perplexity 4.7481
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
fig.suptitle('V1.0 Training Metrics Over Time', fontsize=18, y=0.99)

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