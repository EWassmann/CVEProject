import cv2
import numpy as np
import pyrealsense2
import open3d as o3d
#first part of project, start video stream to realsense camera, get depth frames and the alighned rgb frames, turn those depth frames into a point cloud, mask the point cloud using the rgb info probably (or mask directly on point cloud) so that we are left with only the object
#section output- open3d point cloud objects, maybe masks (np array) of the rgb or depth frames as well


# second part of project, find the transforms between the depth frames (maybe use icp) and if we are unable to mask the point clouds before we find the transforms, apply the masks to the depth images now and create smaller point clouds to apply the same transforms to
#section output- a single point cloud representing the object.

#third part of project, do post processing on the point cloud as needed, turn it into a mesh probably, compare it to a mesh we read in? alighn it with that read in mesh?
#section output- two meshes probably? transforms?

#fourth part of the project, visualize the two meshes, diffrences between them, make good graphics to show off what we did before