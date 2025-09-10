import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def load_voxel_as_pointcloud(vox_path, num_points=1024):
    mat = sio.loadmat(vox_path)
    vox = mat['vox']  # shape [32,32,32] or similar
    points = np.argwhere(vox > 0)  # get occupied voxels
    points = points / np.array(vox.shape)  # normalize to [0,1]

    if points.shape[0] >= num_points:
        idx = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[idx]
    else:
        pad = np.zeros((num_points - points.shape[0], 3))
        points = np.vstack([points, pad])
    return vox, points  # return both

def visualize_voxel_and_pointcloud(vox_path):
    vox, points = load_voxel_as_pointcloud(vox_path)

    fig = plt.figure(figsize=(12,6))

    # Plot voxel grid
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.voxels(vox, edgecolor='k', facecolors='cyan', alpha=0.5)
    ax1.set_title("Voxel Grid")
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    # Plot point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:,0], points[:,1], points[:,2], s=5, c='red')
    ax2.set_title("Point Cloud")
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

# Example usage
visualize_voxel_and_pointcloud('model\bed\bed_0\voxel.mat')
