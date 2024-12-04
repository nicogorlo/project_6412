import numpy as np
import matplotlib.pyplot as plt

def minkowski_sum(set1, set2):
    """
    Compute the Minkowski sum of two sets of points.
    
    Parameters:
        set1: ndarray of shape (N, 2) - points in the first set
        set2: ndarray of shape (M, 2) - points in the second set
    
    Returns:
        ndarray: Points in the Minkowski sum
    """
    return np.array([p1 + p2 for p1 in set1 for p2 in set2])

def plot_set(ax, points, label=None, **kwargs):
    """Helper function to plot a set of points."""
    ax.scatter(points[:, 0], points[:, 1], label=label, **kwargs)
    if label:
        ax.legend()

class ball:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def sample(self, N):
        """Sample N points uniformly from the ball."""
        p = np.random.normal(size=(N, 2))
        p /= np.linalg.norm(p, axis=1)[:, None]
        p *= np.random.uniform(0, self.radius, N)[:, None]
        return p + self.center
    

    

def main():
    b1 = ball(np.array([0, 0]), 1)
    b2 = ball(np.array([1, 1]), 0.5)

    # Sample points from the two balls
    N = 1000
    points1 = b1.sample(N)
    points2 = b2.sample(N)

    # Compute the Minkowski sum
    sum_points = minkowski_sum(points1, points2)

    # Plot the sets and the Minkowski sum
    fig, ax = plt.subplots()
    plot_set(ax, points1, label='Ball 1', c='b', alpha=0.1)
    plot_set(ax, points2, label='Ball 2', c='r', alpha=0.1)
    plot_set(ax, sum_points, label='Minkowski sum', c='g', alpha=0.3)
    ax.set_aspect('equal', 'box')
    plt.show()

if __name__ == '__main__':
    main()