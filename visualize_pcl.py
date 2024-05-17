import open3d as o3d
from open3d import visualization

# Visualize point cloud
def main():

    cloud = o3d.io.read_point_cloud('./ground_truth_downsampled.ply')
    visualization.draw_geometries([cloud])    

if __name__ == "__main__":
    main()