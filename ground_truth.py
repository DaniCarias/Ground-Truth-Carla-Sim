from utils import setup_carla, spawn_vehicle, spawn_cameras, get_intrinsic_extrinsic_matrix, point2D_to_point3D, downsample
import carla
import queue
import numpy as np
import open3d as o3d
import cv2
import argparse


parser = argparse.ArgumentParser(
    prog='Get Point Cloud downsampled (ground truth) from RGBD',
    description='This script gets the point cloud from the RGB and Depth cameras, than downsample it and saves it in a .ply and .pcd files.',
)
parser.add_argument('--frames', '-F', default='20', help='Interval of frames to get the point cloud (default = 20')
parser.add_argument('--pcl', '-P', help='Point Cloud path, if you want to add more pointclouds to an existing Point Cloud file')
parser.add_argument('--leaf_size', '-L', default='0.06', help='Leaf size to downsample the point cloud (default = 5cm)')
parser.add_argument('--interval', '-I', default='20', help='Interval of frames to downsample the point cloud (default = 20 frames)')

arguments = parser.parse_args()


def main():
    actor_list = []
    IMG_WIDTH = 800
    IMG_HEIGHT = 600
    colors = np.empty((0, 3))
    points = np.empty((0, 3))
    

    try:
        
        # Load the point cloud if the user wants to add more point clouds to an existing point cloud
        if arguments.pcl != None:
            pcl_downsampled = o3d.io.read_point_cloud(arguments.pcl)
            print(f"Loaded point cloud from {pcl_downsampled}")
            o3d.visualization.draw_geometries([pcl_downsampled])
        else:
            pcl_downsampled = o3d.geometry.PointCloud()
        
        
        # Setup the world
        world, blueprint_library, traffic_manager = setup_carla()
        
        # Settings
        settings = world.get_settings()
        settings.no_rendering_mode = True # No rendering mode
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Vehicle
        vehicle = spawn_vehicle(world, blueprint_library)
        vehicle.set_autopilot(True)
        traffic_manager.ignore_lights_percentage(vehicle, 100) # Ignore all the red ligths
        
        
        # Depth & RGB FRONT
        camera_transform = carla.Transform(carla.Location(z=1.7))
        camera_depth_front = spawn_cameras('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT, camera_transform)
        camera_rgb_front = spawn_cameras('sensor.camera.rgb', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT, camera_transform)
        # Depth & RGB RIGHT
        camera_transform_right = carla.Transform(carla.Location(z=1.7), carla.Rotation(yaw=90.0))
        camera_depth_right = spawn_cameras('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT, camera_transform_right)
        camera_rgb_right = spawn_cameras('sensor.camera.rgb', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT, camera_transform_right)
        # Depth & RGB LEFT
        camera_transform_left = carla.Transform(carla.Location(z=1.7), carla.Rotation(yaw=-90.0))
        camera_depth_left = spawn_cameras('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT, camera_transform_left)
        camera_rgb_left = spawn_cameras('sensor.camera.rgb', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT, camera_transform_left)
        # Depth & RGB BACK
        camera_transform_back = carla.Transform(carla.Location(z=1.7), carla.Rotation(yaw=180.0))
        camera_depth_back = spawn_cameras('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT, camera_transform_back)
        camera_rgb_back = spawn_cameras('sensor.camera.rgb', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT, camera_transform_back)
        
        actor_list.extend([vehicle, camera_depth_front, camera_rgb_front, camera_depth_right, camera_rgb_right, camera_depth_left, camera_rgb_left, camera_depth_back, camera_rgb_back])

        # Queues
        image_queue_depth_front = queue.Queue()
        image_queue_rgb_front = queue.Queue()
        image_queue_depth_right = queue.Queue()
        image_queue_rgb_right = queue.Queue()
        image_queue_depth_left = queue.Queue()
        image_queue_rgb_left = queue.Queue()
        image_queue_depth_back = queue.Queue()
        image_queue_rgb_back = queue.Queue()
        
        # Listen to the cameras
        camera_depth_front.listen(image_queue_depth_front.put)
        camera_rgb_front.listen(image_queue_rgb_front.put)
        camera_depth_right.listen(image_queue_depth_right.put)
        camera_rgb_right.listen(image_queue_rgb_right.put)
        camera_depth_left.listen(image_queue_depth_left.put)
        camera_rgb_left.listen(image_queue_rgb_left.put)
        camera_depth_back.listen(image_queue_depth_back.put)
        camera_rgb_back.listen(image_queue_rgb_back.put)
        
        tick = -1
        while cv2.waitKey(1) != ord('q'): # Press 'q' in the RGB camera image to stop the script savely
            world.tick()
            tick += 1
            
            image_depth_front = image_queue_depth_front.get()
            rbg_image_front = image_queue_rgb_front.get()
            image_depth_right = image_queue_depth_right.get()
            rbg_image_right = image_queue_rgb_right.get()
            image_depth_left = image_queue_depth_left.get()
            rbg_image_left = image_queue_rgb_left.get()
            image_depth_back = image_queue_depth_back.get()
            rbg_image_back = image_queue_rgb_back.get()
            
            # Get the data with a interval of 'arguments.frames' ticks
            if tick % int(arguments.frames) == 0:
                
                rbg_image_front = np.reshape(np.copy(rbg_image_front.raw_data), (rbg_image_front.height, rbg_image_front.width, 4))
                rbg_image_right = np.reshape(np.copy(rbg_image_right.raw_data), (rbg_image_right.height, rbg_image_right.width, 4))
                rbg_image_left = np.reshape(np.copy(rbg_image_left.raw_data), (rbg_image_left.height, rbg_image_left.width, 4))
                rbg_image_back = np.reshape(np.copy(rbg_image_back.raw_data), (rbg_image_back.height, rbg_image_back.width, 4))
                
                # Show the RGB images
                cv2.imshow('RGB Camera Front Output', rbg_image_front)
                #cv2.imshow('RGB Camera Right Output', rbg_image_right)
                #cv2.imshow('RGB Camera Left Output', rbg_image_left)
                #cv2.imshow('RGB Camera Back Output', rbg_image_back)
                
                intrinsic_matrix_front, camera2world_matrix_front = get_intrinsic_extrinsic_matrix(camera_depth_front, image_depth_front)
                intrinsic_matrix_right, camera2world_matrix_right = get_intrinsic_extrinsic_matrix(camera_depth_right, image_depth_right)
                intrinsic_matrix_left, camera2world_matrix_left = get_intrinsic_extrinsic_matrix(camera_depth_left, image_depth_left)
                intrinsic_matrix_back, camera2world_matrix_back = get_intrinsic_extrinsic_matrix(camera_depth_back, image_depth_back)
                
                
                # Get the points [[X...], [Y...], [Z...]] and the colors [[R...], [G...], [B...]] normalized
                points_3D_front, color_front = point2D_to_point3D(image_depth_front, rbg_image_front[..., [2, 1, 0]], intrinsic_matrix_front)
                points_3D_right, color_right = point2D_to_point3D(image_depth_right, rbg_image_right[..., [2, 1, 0]], intrinsic_matrix_right)
                points_3D_left, color_left = point2D_to_point3D(image_depth_left, rbg_image_left[..., [2, 1, 0]], intrinsic_matrix_left)
                points_3D_back, color_back = point2D_to_point3D(image_depth_back, rbg_image_back[..., [2, 1, 0]], intrinsic_matrix_back)
                
                
                # To multiply by the extrinsic matrix (same shape as the camera2world_matrix matrix)
                p3d_front = np.concatenate((points_3D_front, np.ones((1, points_3D_front.shape[1]))))
                p3d_right = np.concatenate((points_3D_right, np.ones((1, points_3D_right.shape[1]))))
                p3d_left = np.concatenate((points_3D_left, np.ones((1, points_3D_left.shape[1]))))
                p3d_back = np.concatenate((points_3D_back, np.ones((1, points_3D_back.shape[1]))))
                
                # Get the 3D points in the world
                p3d_world_front = np.dot(camera2world_matrix_front, p3d_front)[:3]
                p3d_world_right = np.dot(camera2world_matrix_right, p3d_right)[:3]
                p3d_world_left = np.dot(camera2world_matrix_left, p3d_left)[:3]
                p3d_world_back = np.dot(camera2world_matrix_back, p3d_back)[:3]
                
                # Reshape the array to (height * width, 3) -> X, Y and Z for each point
                p3d_world_front = np.transpose(p3d_world_front)
                p3d_world_right = np.transpose(p3d_world_right)
                p3d_world_left = np.transpose(p3d_world_left)
                p3d_world_back = np.transpose(p3d_world_back)

                colors = np.concatenate((colors, color_front, color_right, color_left, color_back))
                points = np.concatenate((points, p3d_world_front, p3d_world_right, p3d_world_left, p3d_world_back))

                print(f"PointCloud with {p3d_world_front.shape[0] + p3d_world_right.shape[0] + p3d_world_left.shape[0] + p3d_world_back.shape[0]} points")

            if tick % int(arguments.interval) == 0:
                
                print("\n")
                # Downsample the point cloud
                downsampled_points, downsampled_colors = downsample(points, colors, arguments.leaf_size)
                print("\n")
                
                # Clear the points and colors to get the next frame
                colors = np.empty((0, 3))
                points = np.empty((0, 3))

                # Create a point cloud to save the downsampled point cloud of the atually frame
                frame_pcl_downsampled = o3d.geometry.PointCloud()
                frame_pcl_downsampled.points.extend(o3d.utility.Vector3dVector(downsampled_points))
                downsampled_colors = np.clip(downsampled_colors / 255.0, 0, 1)
                frame_pcl_downsampled.colors.extend(o3d.utility.Vector3dVector(downsampled_colors))
                #o3d.visualization.draw_geometries([frame_pcl_downsampled])
                
                
                # Add the downsampled point cloud of the frame to the total point cloud
                pcl_downsampled.points.extend(frame_pcl_downsampled.points)
                pcl_downsampled.colors.extend(frame_pcl_downsampled.colors)


        print("Saving the downsampled point cloud...")
        o3d.io.write_point_cloud(f"./ground_truth_downsampled.ply", pcl_downsampled)

        print(f"---> Total Downsampled {pcl_downsampled}")
        o3d.visualization.draw_geometries([pcl_downsampled]) # If you want to see the total point cloud every 120 frames
        
    finally:

        for actor in actor_list:
            actor.destroy()
        print(f"All cleaned up!")


if __name__ == '__main__':
    main()