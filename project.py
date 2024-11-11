import cv2
import numpy as np
import pyrealsense2
import open3d as o3d
#first part of project, start video stream to realsense camera, get depth frames and the alighned rgb frames, turn those depth frames into a point cloud, mask the point cloud using the rgb info probably (or mask directly on point cloud) so that we are left with only the object
#section output- open3d point cloud objects, maybe masks (np array) of the rgb or depth frames as well
# one of these will probably be useful-
# static create_from_rgbd_image(image, intrinsic, extrinsic=(with default value), project_valid_depth_only=True)
#static create_from_depth_image(depth, intrinsic, extrinsic=(with default value), depth_scale=1000.0, depth_trunc=1000.0, stride=1, project_valid_depth_only=True)
#likely want to do some preprocessing as well 



# second part of project, find the transforms between the depth frames (maybe use icp) and if we are unable to mask the point clouds before we find the transforms, apply the masks to the depth images now and create smaller point clouds to apply the same transforms to
#section output- a single point cloud representing the object.
#apparently the way to do this is to apply global regestration and then use that transform to feed local regestration
# https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
# https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
#  we will likely need to appy ransac ourselves- this is more than enough work



#third part of project, do post processing on the point cloud as needed, turn it into a mesh probably, compare it to a mesh we read in? alighn it with that read in mesh?
#section output- two meshes probably? transforms?

#Three techneques. alpha shapes, ball pivoting, and poisson surface reconstruction.
#also need to do normal vector estimation, and there is the issue with direction in normal vec estimation, unsure if this applies here

#assuming from previous step we have: transfomred merged point clouds

merged_point_clouds = o3d.geometry.PointCloud()
##Alpha shaped
alpha = 0.03 # tradeoff paramater- mess with changing this around
print(f"alpha={alpha:.3f}")
alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(merged_point_clouds, alpha)
alpha_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([alpha_mesh], mesh_show_back_face=True)
##Ball pivoting 
## Next 2 techneques assume that point clouds have normals 
merged_point_clouds.estimate_normals() #there are some diffrent paramaters in this we can change if we are not happy with the output
merged_point_clouds.orient_normals_consistent_tangent_plane(100) #that 100 is the number of points to use in the graph to propogate normal orientation- may want to change

radii = [0.005, 0.01, 0.02, 0.04] #these radii are " a parameter that corresponds to the radii of the individual balls that are pivoted on the point cloud." may want to change
ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(merged_point_clouds, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([merged_point_clouds, ball_mesh])

## Poisson surface reconstruction, APPARENTLY SUPPOSED TO BE GOOD
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    Poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        merged_point_clouds, depth=9) #the larger the depth value is the more detail will be in the mesh
##density value can be visualized to show how many supporting points there are for each vertex
print(Poisson_mesh)
o3d.visualization.draw_geometries([Poisson_mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])

#deoending on quality of mesh we can smooth it, there are several ways to do this. 



#read in mesh of intrest,
input_mesh = o3d.io.read_triangle_mesh("path to mesh, idk if it should be string") #

#sounds like poisson disk sampling is the best way to get a point cloud from the mesh
input_pcd = input_mesh.sample_points_poisson_disk(number_of_points=500, init_factor=5) #may want to mess with these inputs as well
#now lets do point cloud regestration to find the transform, apply to meshes and compare 

#transform just like in part 2
#how should we compare?

#distance between each point and the surface of the mesh? -  then make a color map? probably best idea
#lets try using point cloud distance

dists = merged_point_clouds.compute_point_cloud_distance(input_pcd) #doistance between the point cloud and closest point on the input
## now we just need to find a way to visualize these point cloud distances.




#now how do we alighn the meshes? best way may be to take the input mesh and then get a point cloud from is and use
#global regestration on the two of them.




#fourth part of the project, visualize the two meshes, diffrences between them, make good graphics to show off what we did before