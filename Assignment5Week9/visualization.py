import matplotlib.pyplot as plt

# Data points (steps)
steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999]

# Config 1 (0.83M parameters)
config1_train = [3.7909, 0.3599, 0.3403, 0.3343, 0.3312, 0.3303, 0.3295, 0.3286, 0.3272, 0.3263, 0.3260]
config1_val = [3.7915, 0.3621, 0.3437, 0.3371, 0.3354, 0.3340, 0.3347, 0.3349, 0.3348, 0.3351, 0.3352]

# Config 2 (0.88M parameters)
config2_train = [3.7483, 0.3491, 0.3413, 0.3357, 0.3364, 0.3342, 0.3332, 0.3302, 0.3288, 0.3267, 0.3253]
config2_val = [3.7492, 0.3552, 0.3478, 0.3438, 0.3450, 0.3441, 0.3450, 0.3442, 0.3444, 0.3449, 0.3453]

# Config 3 (0.22M parameters)
config3_train = [3.6824, 0.5153, 0.3515, 0.3408, 0.3369, 0.3343, 0.3331, 0.3305, 0.3309, 0.3297, 0.3299]
config3_val = [3.6835, 0.5194, 0.3528, 0.3430, 0.3389, 0.3373, 0.3361, 0.3341, 0.3343, 0.3333, 0.3338]

plt.figure(figsize=(12, 6))

# Plot training losses
plt.plot(steps, config1_train, '-o', label='Config 1 Train (0.83M params)', alpha=0.7)
plt.plot(steps, config2_train, '-o', label='Config 2 Train (0.88M params)', alpha=0.7)
plt.plot(steps, config3_train, '-o', label='Config 3 Train (0.22M params)', alpha=0.7)

# Plot validation losses
plt.plot(steps, config1_val, '--o', label='Config 1 Val', alpha=0.7)
plt.plot(steps, config2_val, '--o', label='Config 2 Val', alpha=0.7)
plt.plot(steps, config3_val, '--o', label='Config 3 Val', alpha=0.7)

plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Losses for Different Model Configurations')
plt.legend()
plt.grid(True, alpha=0.3)

# Set y-axis to start from 0.3 for better visualization of the relevant range
plt.ylim(0.3, 0.6)

plt.tight_layout()
plt.show()