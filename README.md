# Get Point Cloud ground truth of a map in Carla Sim

3D mapping in Carla Sim using RGB and Depth cameras to obtain the ground truth in a point cloud format.
The point cloud is downsampled using a voxelgrid filter with PCL library.


## Quick Start
### Clone this repository & run the Carla Sim
```
git clone https://github.com/DaniCarias/Ground-Truth-Carla-Sim.git
cd [YOUR-PATH-TO-CARLA]
.\CarlaUE4.sh
```

### Get the ground truth
```
python3 ground_truth.py
```
#### You can define...
* The range of frame to obtain a Point Cloud (default = 20):
```
python3 ground_truth.py -F 30
```
* The Point Cloud that you already have and want to add more points to:
```
python3 ground_truth.py -P [PATH-TO-PCL-FILE]
```
* The leaf size you want to downsample the Point Cloud (default = 0.06 (6cm)):
```
python3 ground_truth.py -L 0.1
```
* The interval of frames to downsample the point cloud (default = 20 frames)
```
python3 ground_truth.py -I 30
```
### To save and view the Point Cloud ground truth
If you want to finish the 3D mapping click on the `"Q"` key to save and view the PCL.

### If you only want to view the Point Cloud file
```
python3 visualize_pcl.py
```

## Notes
Based on this [repo](https://github.com/iaoqian/carla_3d-mapping_and_adding_virtual_objects/tree/main), I created my version with downsampling and some changes to im


