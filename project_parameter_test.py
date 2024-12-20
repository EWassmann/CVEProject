import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import matplotlib.pyplot as plt
import tqdm
import dask
from dask import delayed, compute
import os
import time
#import open3d_tutorial as o3dtut
'''
First part of the project
1. Start video stream to realsense camera
2. Get depth frames and the aligned rgb frames
3. Turn those depth frames into a point cloud
4. Mask the point cloud using the rgb info probably (or mask directly on point cloud) 
so that we are left with only the object section output
    - open3d point cloud objects
    - maybe masks (np array) of the rgb or depth frames as well

One of these will probably be useful-
static create_from_rgbd_image(image, intrinsic, extrinsic=(with default value), project_valid_depth_only=True)
static create_from_depth_image(depth, intrinsic, extrinsic=(with default value), depth_scale=1000.0, depth_trunc=1000.0, stride=1, project_valid_depth_only=True)

Likely want to do some preprocessing as well 
'''
'''
# Reference:
# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py
# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
# https://snyk.io/advisor/python/pyrealsense2/functions/pyrealsense2.pipeline
# https://github.com/AoLyu/Some-implementions-with-RGBD-camera-RealSense-D435/blob/62364d118cef39bf01f6b7bdc7a4ef3edaacf57a/Basic/captureRGBDpt.py#L102
# https://medium.com/@christhaliyath/lidar-and-pointcloud-simple-tutorial-for-op-3c5d7cd35ad4


# try:
#     # Create a pipline to handle the realsense camera
#     pipeline = rs.pipeline()
    
#     # Configure video streams
#     config = rs.config()
    
#     # Capture both depth and color data 
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
#     # Start streaming and obtain camera intrinsics
#     profile = pipeline.start(config)
#     depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
#     intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
#     print((intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy))
#     pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
#     fileNames = []
#     # Capture frames
#     for i in range(10): # need to adjust the frame counts based on our use case
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
        
#         if (depth_frame == None) or (color_frame == None):
#             print("Depth frames or color frames are not captured.\n")
#             continue
        
#         # Obtain Open3D images for depth, color, and RGBD
#         depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
#         color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
#         rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, convert_rgb_to_intensity=False)
        
#         # Generate point clouds
#         point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_intrinsics)
#         point_cloud.estimate_normals()
#         # To standardize coordinate system
#         matrix = [
#             [1, 0, 0, 0],  # X - Remains the same
#             [0, -1, 0, 0], # Y - Inverted
#             [0, 0, -1, 0], # Z - Inverted
#             [0, 0, 0, 1]   # No translation
#         ]
#         point_cloud.transform(matrix)
        
#         # Need to be modified based on depth (in meters)
#         threshold = 0.3
#         point_cloud_points = np.asarray(point_cloud.points)
#         point_cloud_rgb = np.asarray(point_cloud.colors)
        
#         # Create a mask for point cloud (z-axis)
#         mask = point_cloud_points[:, 2] < threshold
        
#         # Filter out points and RGB
#         filtered_points = point_cloud_points[mask]
#         filtered_rgb = point_cloud_rgb[mask]
        
#         # Visualize masked point cloud 
#         point_cloud_masked = o3d.geometry.PointCloud()
#         point_cloud_masked.points = o3d.utility.Vector3dVector(filtered_points)
#         point_cloud_masked.colors = o3d.utility.Vector3dVector(filtered_rgb)
#         point_cloud_masked.estimate_normals()
#         #o3d.visualization.draw_geometries([point_cloud_masked])
#         st = "masked_point_cloud"+str(i)+".pcd"
#         fileNames.append(st)
#         # Save the masked point cloud
#         o3d.io.write_point_cloud(st, point_cloud_masked)
#         #time.sleep(0.5)
#         input("press enter")
        
# # For debugging purposes
# except Exception as e:
#     print(e)
#     pass
'''
#fileNames = ["saved_pct/masked_point_cloud0.pcd","saved_pct/masked_point_cloud1.pcd","saved_pct/masked_point_cloud2.pcd","saved_pct/masked_point_cloud3.pcd","saved_pct/masked_point_cloud4.pcd","saved_pct/masked_point_cloud5.pcd","saved_pct/masked_point_cloud6.pcd","saved_pct/masked_point_cloud7.pcd","saved_pct/masked_point_cloud8.pcd","saved_pct/masked_point_cloud9.pcd",]
#fileNames = ["masked_point_cloud0.pcd","masked_point_cloud1.pcd","masked_point_cloud2.pcd","masked_point_cloud3.pcd","masked_point_cloud4.pcd","masked_point_cloud5.pcd","masked_point_cloud6.pcd","masked_point_cloud7.pcd","masked_point_cloud8.pcd","masked_point_cloud9.pcd","masked_point_cloud10.pcd","masked_point_cloud11.pcd","masked_point_cloud12.pcd","masked_point_cloud13.pcd","masked_point_cloud14.pcd"]
'''fileNames = []
for i in range(0,15):
    fileNames.append("D:\\Courses\\CMU-ComputerVisionForEng\\00 PROJ\\saved_pct\\masked_point_cloud"+str(i)+".pcd")
print("1")'''

# second part of project, find the transforms between the depth frames (maybe use icp) and if we are unable to mask the point clouds before we find the transforms, apply the masks to the depth images now and create smaller point clouds to apply the same transforms to
#section output- a single point cloud representing the object.
#apparently the way to do this is to apply global regestration and then use that transform to feed local regestration
# https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
# https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
#  we will likely need to appy ransac ourselves- this is more than enough work


def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh
'''

# mesh_rabbit_info = o3d.data.BunnyMesh() #o3d.get_bunny_mesh()
# #print(mesh_rabbit_info.path)
# mesh_rabbit=o3d.io.read_triangle_mesh(mesh_rabbit_info.path)
# mesh_rabbit.compute_vertex_normals()
# #o3d.visualization.draw_geometries([mesh_rabbit])
# pcd_rabbit = mesh_rabbit.sample_points_poisson_disk(750)
# pcd_rabbit_transformed = mesh_rabbit.sample_points_poisson_disk(1000)
# #o3d.visualization.draw_geometries([pcd_rabbit_transformed])
# print("1.1")
# t_M = np.asarray([[1,0,0,0.1],
#                 [0,np.cos(45*np.pi/180),np.sin(45*np.pi/180),0],
#                 [0,-np.sin(45*np.pi/180),np.cos(45*np.pi/180),.1],
#                 [0,0,0,1]])
# T = t_M.astype(np.float64)

# T = T.astype(np.float64)
# # T[:3, :3] = mesh_rabbit.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
# # T[0, 3] = 1
# # T[1, 3] = 1.3
# print(T)
# print("1.2")
# pcd_rabbit_transformed.transform(T)
# o3d.visualization.draw_geometries([pcd_rabbit,pcd_rabbit_transformed])
# print("1.25")


'''
# Downsamples pointcloud source and target 
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
print("1.5")


# Runs RANSAC on downsampled pointclouds
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, iteration):
    distance_threshold = voxel_size * 2.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
         o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.99),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
         o3d.pipelines.registration.RANSACConvergenceCriteria(iteration, 0.999))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

voxel_size = 0.03  # means 1mm for this dataset
threshold = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(fileNames):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    for source_id in range(len(fileNames)):
            #o3d.visualization.draw_geometries([o3d.io.read_point_cloud(fileNames[source_id])])
            for target_id in range(source_id + 1,len(fileNames) ):
                #o3d.io.read_point_cloud(fileNames[source_id]).orient_normals_consistent_tangent_plane()
                
                transformation_icp, information_icp = pairwise_registration(
                    o3d.io.read_point_cloud(fileNames[source_id]),o3d.io.read_point_cloud(fileNames[target_id]))
                print("Build o3d.pipelines.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=True))
    return pose_graph



'''
pose_graph = full_registration(fileNames)

option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)


pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(fileNames)):
    pcd = o3d.io.read_point_cloud(fileNames[point_id])
    pcd.transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcd
#pcd_combined_down = pcd_combined
pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
#o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
o3d.io.write_point_cloud("D:\\Courses\\CMU-ComputerVisionForEng\\00 PROJ\\multiway_registration.pcd", pcd_combined_down)
o3d.visualization.draw_geometries([pcd_combined_down],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])'''



'''
print("2")
source = o3d.io.read_point_cloud(fileNames[0])
for file in fileNames: # loop through frames in rotation order, or use an evaluation metric to compare two frames and make sure they have enough overlap
    if(file != fileNames[0]):
        voxel_size = 0.01  # means 1cm for this dataset
        threshold = 0.02
        # Loop through each frame
        target = o3d.io.read_point_cloud(file) # set to base depth frame

        # evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
        # print(evaluation)

        # Take data set and downsample point cloud to prepare for RANSAC
        #source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
        # Obtain rough tansformation matrix from RANSAC
        result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh, voxel_size)
        # Obtain refined transformation matrix from ICP, using RANSAC matrix as initial guess
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, result_ransac.transformation,o3d.pipelines.registration.TransformationEstimationPointToPoint())
        source = source + target.transform(np.linalg.inv(reg_p2p.transformation))
        print("merged one")
'''
print("3")

#third part of project, do post processing on the point cloud as needed, turn it into a mesh probably, compare it to a mesh we read in? alighn it with that read in mesh?
#section output- two meshes probably? transforms?

#Three techneques. alpha shapes, ball pivoting, and poisson surface reconstruction.
#also need to do normal vector estimation, and there is the issue with direction in normal vec estimation, unsure if this applies here

#assuming from previous step we have: transfomred merged point clouds
pcd_combined_down = o3d.io.read_point_cloud("D:\\Courses\\CMU-ComputerVisionForEng\\00 PROJ\\multiway_registration.pcd")
#pcd_combined_down = o3d.io.read_point_cloud("multiway_registration.pcd")
merged_point_clouds = pcd_combined_down

#show the point cloud
#o3d.visualization.draw_geometries([merged_point_clouds])

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        merged_point_clouds.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))

# Find the number of clusters
max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")

# Separate clusters into different point clouds
cluster_point_clouds = []
for cluster_idx in range(max_label + 1):
    # Select points that belong to the current cluster
    cluster_points = np.asarray(merged_point_clouds.points)[labels == cluster_idx]
    cluster_colors = np.asarray(merged_point_clouds.colors)[labels == cluster_idx]

    # Create a new point cloud for this cluster
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

    # Append the cluster point cloud to the list
    cluster_point_clouds.append(cluster_pcd)

merged_point_clouds = cluster_point_clouds[0]

table_pcd = cluster_point_clouds[1]
plane_model, inliers = table_pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model



# inlier_cloud = table_pcd.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud = table_pcd.select_by_index(inliers, invert=True)
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                                   zoom=0.8,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])




def plane_to_z_transform(A, B, C, D):
    # Normalize the normal vector
    normal = np.array([A, B, C])
    norm = np.linalg.norm(normal)
    normal_unit = normal / norm

    # Translation vector to move the plane to the origin
    translation = -(D / norm) * normal_unit

    # Rotation to align the normal vector with the z-axis
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal_unit, z_axis)  # Axis of rotation
    s = np.linalg.norm(v)             # Magnitude of the rotation vector
    c = np.dot(normal_unit, z_axis)   # Cosine of the angle
    v_skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])  # Skew-symmetric matrix of v

    if s == 0:  # Already aligned, no rotation needed
        R = np.eye(3)
    else:
        R = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / s**2)

    # Combine rotation and translation into a 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T

# Example plane equation: 2x + 3y + 4z - 5 = 0

T = plane_to_z_transform(a, b, c, d)
print("Transformation Matrix:\n", T)

# Apply the transformation to a point cloud

table_pcd.transform(np.linalg.inv(T))

pcd_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
#o3d.visualization.draw_geometries([table_pcd,pcd_frame])



# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# merged_point_clouds.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([merged_point_clouds],
#                                   zoom=0.455,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])





# o3d.visualization.draw_geometries([merged_point_clouds,pcd_frame])

# o3d.visualization.draw_geometries([source])

# ##Alpha shaped
# alpha = 0.03 # tradeoff paramater- mess with changing this around
# print(f"alpha={alpha:.3f}")
# alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(merged_point_clouds, alpha)
# alpha_mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([alpha_mesh], mesh_show_back_face=True)
# ##Ball pivoting 

# ## the techneques assume that point clouds have normals 
merged_point_clouds.estimate_normals() 
merged_point_clouds.orient_normals_consistent_tangent_plane(100) 
#that 100 is the number of points to use in the graph to propogate normal orientation- may want to change

# radii = [0.005, 0.01, 0.02, 0.04] #these radii are " a parameter that corresponds to the radii of the individual balls that are pivoted on the point cloud." may want to change
# ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(merged_point_clouds, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([merged_point_clouds, ball_mesh])

# Poisson surface reconstruction
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    Poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        merged_point_clouds, depth=9) #the larger the depth value is the more detail will be in the mesh
    
#density value can be visualized to show how many supporting points there are for each vertex
reconstructed_mesh = Poisson_mesh
reconstructed_mesh.compute_vertex_normals()


vertex_normals = np.asarray(reconstructed_mesh.vertex_normals)
flipped_vertex_normals = -vertex_normals  # Flip orientation
reconstructed_mesh.vertex_normals = o3d.utility.Vector3dVector(flipped_vertex_normals)

#reconstructed_mesh.orient_normals_consistent_tangent_plane()
# reconstructed_mesh.compute_triangle_normals()
reconstructed_mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
# o3d.visualization.draw_geometries([reconstructed_mesh]) 
o3d.io.write_triangle_mesh("reconstructed_mesh.stl",reconstructed_mesh)
# o3d.visualization.draw_geometries([Poisson_mesh],
#                                   zoom=0.664,
#                                   front=[-0.4761, -0.4698, -0.7434],
#                                   lookat=[1.8900, 3.2596, 0.9284],
#                                   up=[0.2304, -0.8825, 0.4101])

#deoending on quality of mesh we can smooth it, there are several ways to do this. 



#read in mesh of intrest,
#input_mesh = o3d.io.read_triangle_mesh("STL_Files\Beam_as_built_cm.stl") 
input_mesh = o3d.io.read_triangle_mesh("D:\\Courses\\CMU-ComputerVisionForEng\\00 PROJ\STL_Files\\Beam_as_built_cm.stl")
# input_mesh = mesh_rabbit
#poisson disk sampling is the best way 
input_pcd = input_mesh.sample_points_poisson_disk(number_of_points=6400, init_factor=5) 

#input_pcd.transform(np.linalg.inv(T))


#voxel_size = 0.1  # means 1cm for this dataset
#threshold = .85
# Loop through each frame
target = input_pcd # set to base depth frame
source = merged_point_clouds
# evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
# print(evaluation)

# Function to process a single parameter combination
@delayed
def process_combination(voxel_size, voxel_size_down, threshold, max_it, max_it_ransac, rmse, input_pcd, merged_point_clouds):
    # Adjust parameter values
    voxel_size_down *= 0.2
    threshold *= 0.2
    max_it = max_it * 100000
    max_it_ransac = max_it_ransac * 100000
    rmse = pow(1, rmse - 10)
    source = input_pcd
    target = merged_point_clouds

    # Prepare dataset and execute RANSAC
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size_down)
    result_ransac = execute_global_registration(source, target, source_fpfh, target_fpfh, voxel_size, max_it_ransac)

    # Perform ICP for refinement
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_it, relative_rmse=rmse)
    )

    # Transform source and compute distances
    source.transform(reg_p2p.transformation)
    merged_point_clouds = source

    dists = merged_point_clouds.compute_point_cloud_distance(target)
    avg_local = np.mean(dists)
    max_local = np.max(dists)

    return avg_local, max_local, dists

# Main function
if __name__ == "__main__":
    avg = []
    max = []

    # Placeholder point clouds (replace with actual data)
    #source = merged_point_clouds  # Example source point cloud
    #target = input_pcd  # Example target point cloud
    #input_pcd = o3d.geometry.PointCloud()  # Example input point cloud

    # Generate all parameter combinations
    parameter_combinations = [
        (voxel_size * 0.2, voxel_size_down, threshold, max_it, max_it_ransac, rmse)
        for voxel_size in range(1, 5)
        for voxel_size_down in range(1, 5)
        for threshold in range(1, 5)
        for max_it in range(1, 5)
        for max_it_ransac in range(1, 5)
        for rmse in range(1, 5)
    ]

    # Create Dask tasks for all parameter combinations
    tasks = [
        process_combination(
            voxel_size, voxel_size_down, threshold, max_it, max_it_ransac, rmse,
            input_pcd, merged_point_clouds
        )
        for voxel_size, voxel_size_down, threshold, max_it, max_it_ransac, rmse in tqdm.tqdm(parameter_combinations)
    ]

    # Execute all tasks in parallel
    results = compute(*tasks)

    # Extract results
    for avg_local, max_local, dists in results:
        avg.append(avg_local)
        max.append(max_local)

    print("Average distances:", avg)
    print("Maximum distances:", max)

#now how do we alighn the meshes? best way may be to take the input mesh and then get a point cloud from is and use
#global regestration on the two of them.


print("5")

#fourth part of the project, visualize the two meshes, diffrences between them, make good graphics to show off what we did before

#%% Colored Difference 3D Point Cloud

'''# Set a threshold for min difference
dists = np.asarray(dists)
threshold = 0.01
# difference_mask = dists > threshold
ind = np.where(dists > 0.001)[0]

# Extract points and distances that differ significantly
difference_points = merged_point_clouds.select_by_index(ind)

difference_colors = np.zeros((np.asarray(difference_points.points).shape[0], 3))  # initialize colors array

# Map distances to colors for visualization (e.g., red for large differences)
max_distance = dists.max()
difference_colors[:, 0] = dists[ind] / max_distance  # Red intensity based on difference

# Create a new point cloud with highlighted differences
diff_pcd = difference_points #o3d.geometry.PointCloud()
#diff_pcd.points = o3d.utility.Vector3dVector(difference_points)
diff_pcd.colors = o3d.utility.Vector3dVector(difference_colors)

# Visualize the difference point cloud
input_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([input_mesh,diff_pcd])
print("6")'''
#%% Layer View
'''
# Convert point clouds to numpy arrays
points1 = np.asarray(merged_point_clouds.points)
points2 = np.asarray(input_pcd.points)

# Calculate half of the maximum z height across both point clouds
max_z = max(points1[:, 2].max(), points2[:, 2].max())
target_z = max_z / 2

# Set a tolerance range for slicing 
tolerance = 0.005
section_points1 = points1[(points1[:, 2] > target_z - tolerance) & (points1[:, 2] < target_z + tolerance)]
section_points2 = points2[(points2[:, 2] > target_z - tolerance) & (points2[:, 2] < target_z + tolerance)]

# Plot the 2D section view for the specified z-height
plt.figure(figsize=(10, 8))

# Plot points from merged_point_clouds in red
plt.scatter(section_points1[:, 0], section_points1[:, 1], color='red', label='Merged Point Cloud', s=5)

# Plot points from input_pcd in blue
plt.scatter(section_points2[:, 0], section_points2[:, 1], color='blue', label='Input Point Cloud', s=5)

# Add labels and legend
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title(f"2D Section View at Z = {target_z:.2f}")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
'''
#%% Comparison Results

# save the average and maximum distances to a text file
with open("D:\\Courses\\CMU-ComputerVisionForEng\\00 PROJ\\comparison_results.txt", "w") as f:
    f.write("Average Distances\n")
    for avg_local in avg:
        f.write(f"{avg_local:.3f}\n")
    f.write("\nMaximum Distances\n")
    for max_local in max:
        f.write(f"{max_local:.3f}\n")




