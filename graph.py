import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('output.txt')

dimension = data[:, 0]
probabilities = data[:, 1:]

dimension_grid, interval_grid = np.meshgrid(dimension, np.arange(100))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(dimension_grid, interval_grid, probabilities.T, cmap='viridis')

ax.set_xlabel('Dimension #')
ax.set_ylabel('Interval (1-100)')
ax.set_zlabel('Probability')
plt.title('CS447 3D Surface Plot')

plt.show()
