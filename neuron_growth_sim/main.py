import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Define the Dendrite class
class Dendrite:
    def __init__(self, x, y, angle, length):
        self.x = x  # Start x-coordinate
        self.y = y  # Start y-coordinate
        self.angle = angle  # Growth direction
        self.length = length  # Branch length
        self.children = []  # Child branches

    def grow(self, growth_rate=1, branch_probability=0.1, pruning_probability=0.02):
        """
        Simulates growth, branching, and pruning of a dendrite.
        """
        # Elongate the branch
        self.length += growth_rate

        # Determine the new endpoint of the branch
        new_x = self.x + self.length * np.cos(self.angle)
        new_y = self.y + self.length * np.sin(self.angle)

        # Branching
        if random.random() < branch_probability:
            branch_angle_variation = np.pi / 4  # ±45° branch angle variance
            left_angle = self.angle + branch_angle_variation
            right_angle = self.angle - branch_angle_variation
            left_branch = Dendrite(new_x, new_y, left_angle, self.length / 2)
            right_branch = Dendrite(new_x, new_y, right_angle, self.length / 2)
            self.children.append(left_branch)
            self.children.append(right_branch)

        # Pruning
        if random.random() < pruning_probability:
            return None  # Prune the branch

        # Update this branch
        return Dendrite(new_x, new_y, self.angle, self.length)

    def plot(self, ax):
        """
        Visualizes the dendrite and its branches recursively.
        """
        end_x = self.x + self.length * np.cos(self.angle)
        end_y = self.y + self.length * np.sin(self.angle)
        ax.plot([self.x, end_x], [self.y, end_y], 'k', linewidth=1)
        for child in self.children:
            child.plot(ax)

# Simulation function
def simulate_growth(steps=20):
    """
    Simulates the growth of a dendritic tree for visualization.
    """
    root = Dendrite(0, 0, np.pi / 2, 1)  # Start with a primary branch pointing up
    all_branches = [root]
    snapshots = []

    for _ in range(steps):
        new_branches = []
        for branch in all_branches:
            if branch is not None:
                grown = branch.grow()
                if grown:
                    new_branches.append(grown)
                new_branches.extend(branch.children)
        all_branches = new_branches
        snapshots.append([b for b in all_branches if b is not None])  # Save state
    
    return root, snapshots

# Initialize simulation
steps = 30
root, snapshots = simulate_growth(steps)

# Animation setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('equal')
ax.set_title("Simulated Purkinje Cell Dendrite Growth")

def update(frame):
    """
    Update function for the animation.
    """
    ax.clear()
    ax.axis('equal')
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 60)
    ax.set_title("Simulated Purkinje Cell Dendrite Growth")
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    
    # Plot all branches at this frame
    branches = snapshots[frame]
    for branch in branches:
        branch.plot(ax)

# Create the animation
anim = FuncAnimation(fig, update, frames=steps, interval=200, repeat=False)

plt.show()
