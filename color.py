import numpy as np

def interpolate_color(start_color, end_color, factor: float):
    """Interpolates between two RGB colors. Factor is between 0 and 1."""
    return tuple(start_color[i] + (end_color[i] - start_color[i]) * factor for i in range(3))

# Example colors
start_color = (232, 130, 103)  # Assuming this is for 32
end_color = (0, 0, 139)  # Deep blue for 131072

# Number of color steps
num_steps = int(np.log2(131072) - np.log2(32))

colors = {2**i: interpolate_color(start_color, end_color, (np.log2(2**i) - np.log2(32)) / num_steps) 
          for i in range(5, 18)}

for number, color in colors.items():
    print(f'{number}: {color},')