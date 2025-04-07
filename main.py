from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
plt.tight_layout()

np.random.seed(4)
centers = [[0, 2], [-0.3, 3.5], [0.5, 1.5]]
k = len(centers)
points, labels = make_blobs(
    n_samples=200,
    centers=centers,
    cluster_std=0.6,
    random_state=4
)

axes[0].scatter(points[:, 0], points[:, 1], c=labels, alpha=0.8)
axes[0].set_title("Source data")

def kmeans_plusplus_init(points, k):
    centers = [points[np.random.choice(len(points))]]
    for _ in range(1, k):
        distances = np.min([np.linalg.norm(points - center, axis=1) for center in centers], axis=0)
        probabilities = distances / np.sum(distances)
        new_center = points[np.random.choice(len(points), p=probabilities)]
        centers.append(new_center)
    return np.array(centers)

def assign_clusters(points, centers):
    distances = np.linalg.norm(points[:, np.newaxis, :] - centers, axis=2)
    cluster_indices = np.argmin(distances, axis=1)
    return cluster_indices

def kmeans(k, points, max_steps=500, threshold=1e-4):
    start_points = kmeans_plusplus_init(points, k)
    step = 0

    while True:
        cluster_indices = assign_clusters(points, start_points)

        axes[1].clear()
        axes[1].set_title(f"Iteration №{step+1}")
        axes[1].scatter(points[:, 0], points[:, 1], c=cluster_indices, alpha=0.8)
        axes[1].scatter(start_points[:, 0], start_points[:, 1], color='red', marker="x")
        plt.pause(0.1)
        fig.canvas.draw()

        new_start_points = np.array([
            np.mean(points[cluster_indices == i], axis=0) if np.any(cluster_indices == i) else start_points[i]
            for i in range(k)
        ])

        if np.linalg.norm(new_start_points - start_points) < threshold or step >= max_steps:
            break
        start_points = new_start_points
        step += 1

kmeans(k, points, max_steps=100)
plt.ioff()
plt.show()
