#! /usr/bin/python3

import time
import rospy
from std_msgs.msg import Header
#import actionlib
#import actionlib_tutorials.msg
#import ensenso_camera_msgs.msg
import numpy as np
import ensenso_camera.ros2 as ros2py
import open3d as o3d
from sensor_msgs.msg._PointCloud2 import PointCloud2
import ros_numpy.src.ros_numpy.point_cloud2 as pc
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
import tf
from sklearn.cluster import spectral_clustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import re
import os
from pyassimp import *
from scipy.spatial.transform import Rotation
import pyvista as pv

RequestData = ros2py.import_action("ensenso_camera_msgs", "RequestData")


def separate_xyz_fromstring(pose):
    pose_list_T = pose.split("rpy")[0]
    pose_list_R = pose.split("rpy")[1]
    T = pose_list_T.split(": ")[1]
    R = pose_list_R.split(": ")[1]

    i = filter(str.isdigit, T)

    xyz_digits = re.findall(r"[-+]?(?:\d*\.*\d+)", T)
    rpy_digits = re.findall(r"[-+]?(?:\d*\.*\d+)", R)

    xyz = np.asarray(xyz_digits, dtype=float)
    rpy = np.asarray(rpy_digits, dtype=float)
    return xyz, rpy


# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8


def convert_rgbUint32_to_tuple(rgb_uint32): return (
    (rgb_uint32 & 0x00ff0000) >> 16, (rgb_uint32 &
                                      0x0000ff00) >> 8, (rgb_uint32 & 0x000000ff)
)


def convert_rgbFloat_to_tuple(rgb_float): return convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)


def convertCloudFromRosToOpen3d(ros_cloud):

    # Get cloud data from ros_cloud
    field_names = [field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(
        ros_cloud, skip_nans=True, field_names=field_names))

    # Check empty
    open3d_cloud = o3d.geometry.PointCloud()
    if len(cloud_data) == 0:
        print("Converting an empty cloud")
        return None

    # Set open3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD = 3  # x, y, z, rgb

        # Get xyz
        xyz = [(x, y, z) for x, y, z, rgb in cloud_data]

        # Get rgb
        # Check whether int or float
        # if float (from pcl::toROSMsg)
        if type(cloud_data[0][IDX_RGB_IN_FIELD]) == float:
            rgb = [convert_rgbFloat_to_tuple(rgb)
                   for x, y, z, rgb in cloud_data]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb)
                   for x, y, z, rgb in cloud_data]

        # combine
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x, y, z) for x, y, z in cloud_data]  # get xyz
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
    


    return open3d_cloud

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)

def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="Workspace"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points = np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors:  # XYZ only
        fields = FIELDS_XYZ
        cloud_data = points
    else:  # XYZ + RGB
        fields = FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255)  # nx3 matrix
        colors = colors[:, 0] * BIT_MOVE_16 + \
            colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
        cloud_data = np.c_[points, colors]

    return pc2.create_cloud(header, fields, cloud_data)

def align_point_cloud(pointcloud, pose):
    xyz = separate_xyz_fromstring(pose)[0]
    rpy = separate_xyz_fromstring(pose)[1]
    # 3D Transformations
    # Transform_matrix = np.arange(xyz, rpy).reshape(4,4)
    # this will not work: you may want to use https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    ################# Using scipy######################
    Rot_mat = Rotation.from_euler('XYZ', rpy, degrees=False).as_matrix()
    Trans_mat = xyz

    Transformed_matrix = np.empty((4, 4))
    Transformed_matrix[:3, :3] = Rot_mat
    Transformed_matrix[:3, 3] = Trans_mat
    Transformed_matrix[3, :] = [0, 0, 0, 1]
    
    Tinv = np.linalg.inv(Transformed_matrix)
    pcd = convertCloudFromRosToOpen3d(pointcloud)
    aligned_pc = pcd.transform(Tinv)
    publish_pose(Tinv, "tool0")
    #o3d.visualization.draw_geometries([aligned_pc])
    return aligned_pc, Tinv


def pose_callback(data):
    rospy.loginfo(data)


def read_file(file_name):
    file_handle = open(file_name)
    pose = file_handle.read()
    print("pose = ", pose)
    file_handle.close()
    return pose


def getpose(i):
    # Read the text file
    file_dir = os.path.realpath('/home/Mikado_FileCam/Part_0_differentGrasp/')
    print(file_dir)
    file_path = 'pose' + str(i) + '.txt'
    file_name = os.path.join(file_dir, file_path)
    pose = read_file(file_name)
    # with open('/home/optonic-shared-pc/Documents/Work/Mikado_FileCam/Part_1/All.txt', 'a') as f:
    #     f.write(str(i))
    #     f.write('\n')
    #     f.write(pose)
    #     f.write('\n')
    return pose


def read_PointCloud(node):
    goal = RequestData.Goal()
    goal.include_results_in_response = True
    goal.request_point_cloud = ros2py.get_param(node, "point_cloud", True)
    goal.publish_results = True
    #goal.request_normals = True
    request_data_client_name = "request_data"
    request_data_client = ros2py.create_action_client(
        node, request_data_client_name, RequestData)

    ros2py.wait_for_server(node, request_data_client, timeout_sec=10)
    response = ros2py.send_action_goal(node, request_data_client, goal)

    if not response.successful():
        node.get_logger().warn("Action was not successful.")
        return
    node.get_logger().info("Action was successful =)")
    result = response.get_result() 
    if result.error.code != 0:
        node.get_logger().error(ros2py.format_error(result.error))
    return result.point_cloud

def create_bounding_box(aligned_pc):
    min_bound = np.array([-.2,-.2,0.52])*.4
    max_bound = np.array([.2,.2,.7])*.4
    # min_bound = np.array([-.2,-.1,0.53])*.38
    # max_bound = np.array([.2,.2,.9])*.4
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcd_crop = aligned_pc.crop(bounding_box)
    if pcd_crop is None:
        print("Aligned pc is none")
    return pcd_crop

def publish_pose(Transform, frame_id):
    
    br = tf.TransformBroadcaster()
    Rot_mat = Transform[:3, :3]
    Trans_mat = Transform[:3, 3]
    br.sendTransform((Trans_mat[0], Trans_mat[1], Trans_mat[2]), Rotation.from_matrix(Rot_mat).as_quat(), rospy.Time.now(), "Workspace", frame_id)


def display_inlier_outlier(cloud, ind): 
    inlier_cloud = cloud.select_by_index(ind)     
    outlier_cloud = cloud.select_by_index(ind, invert=True) 
    print("Inlier: ", inlier_cloud)
    print("Outlier: ", outlier_cloud)             
    print("Showing outliers (red) and inliers (gray): ")   


def filter_outliers(pc):
    voxel_down_pcd = pc.voxel_down_sample(voxel_size=0.002)#(voxel_size=0.025)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=0.5)
    print("ind = ", ind)
    inlier_cloud = cl.select_by_index(ind)
    print("After inlier selection:")
    display_inlier_outlier(voxel_down_pcd, ind)
    return inlier_cloud

def align_point_cloud_gripper_l(pointcloud):
    Rot_mat = Rotation.from_euler('XYZ', np.array([0,0,0]), degrees=False).as_matrix()
    Transformed_matrix = np.empty((4, 4))
    Transformed_matrix[:3, :3] = Rot_mat
    Transformed_matrix[:3, 3] = np.array([0,0.009,-0.22])*.4
    Transformed_matrix[3, :] = [0, 0, 0, 1]
    Tinv = np.linalg.inv(Transformed_matrix)
    aligned_pc = pointcloud.transform(Tinv)
    return aligned_pc

def align_point_cloud_gripper_r(pointcloud):
    Rot_mat = Rotation.from_euler('XYZ', np.array([0,0,3.14159]), degrees=False).as_matrix()
    Transformed_matrix = np.empty((4, 4))
    Transformed_matrix[:3, :3] = Rot_mat
    Transformed_matrix[:3, 3] = np.array([0,0.009,-.22])*.4
    Transformed_matrix[3, :] = [0, 0, 0, 1]
    Tinv = np.linalg.inv(Transformed_matrix)
    aligned_pc = pointcloud.transform(Tinv)
    return aligned_pc

if __name__ == '__main__':

    node = ros2py.create_node("mikado_shreya")
    rospy.loginfo("Inside node")
    pointcloud_list = []
    pose_list = []
    pc = None

    pub = rospy.Publisher("pc_fromMikadoShreya", PointCloud2, queue_size=10)
    T = []
    n_filecams = 20
    rate = rospy.Rate(10)
    for i in range(0, n_filecams, 1):
        print("\n i = ", i)
        pose = getpose(i)
        point_cloud = read_PointCloud(node)
        pub.publish(point_cloud)
        transformed_pc, T = align_point_cloud(point_cloud, pose)
        cropped_pcd = create_bounding_box(transformed_pc)
        if i == 0:
            merged_point_cloud = cropped_pcd
        else:
            merged_point_cloud.points.extend(cropped_pcd.points)
        if merged_point_cloud is None:
            print("merged_point_cloud is None")
        # o3d.visualization.draw_geometries([merged_point_cloud])
        # print("Wait for input")
        # input("Press Enter to continue...")
        # _ = input()
    print("Merge point cloud")    
    #o3d.visualization.draw_geometries([merged_point_cloud])
    pcd = convertCloudFromOpen3dToRos(merged_point_cloud)
    pub.publish(pcd)
    
    ########################################################
    ##Cropping object from gripper
    ########################################################
    
    mesh = o3d.io.read_triangle_mesh("/home/catkin_ws/src/node1/Finger_15cm.stl")
    gripper_pointcloud_l = mesh.sample_points_poisson_disk(100000)
    gripper_pointcloud_r = mesh.sample_points_poisson_disk(100000)
    

    #Make the transformation for gripper so that the gripper is aligned with thc current point cloud 
    transformed_gripper_l = align_point_cloud_gripper_l(gripper_pointcloud_l)
    transformed_gripper_r = align_point_cloud_gripper_r(gripper_pointcloud_r)
    
    #Calculate the difference in point cloud to get the object
    #Cut right finger from the pc
    dists2 = merged_point_cloud.compute_point_cloud_distance(transformed_gripper_r)
    dists2 = np.asarray(dists2)
    ind2 = np.where(dists2 > 0.005)[0]
    print("ind2 = ", ind2)
    pcd_without_right_finger = merged_point_cloud.select_by_index(ind2)
    
    # Cut left finger from the pc
    dists1 = pcd_without_right_finger.compute_point_cloud_distance(transformed_gripper_l)
    dists1 = np.asarray(dists1)
    ind1 = np.where(dists1 > 0.005)[0]
    print("ind1 = ", ind1)
    final_pcd = pcd_without_right_finger.select_by_index(ind1)  
    print("Final pcd")
    o3d.visualization.draw_geometries([final_pcd])

    pcd = convertCloudFromOpen3dToRos(final_pcd)
    pub.publish(pcd)
    
    ###############################
    ##Outlier removal
    ###############################

    ##Filter outliers
    # final_pcd = filter_outliers(final_pcd)
    # # print("####### After filter_outliers #######")
    # o3d.visualization.draw_geometries([final_pcd])

    
    result = o3d.io.write_point_cloud('/home/Output_STL_Files/Part_0_differentGrasp/object1_pcd.pcd', final_pcd)
    print("File saved in .pcd format = ", result)
    ###############################
    ##Mesh reconstruction
    ###############################
    
    ##(1) Alpha shape
    ###################################################################
    alpha=0.004
    print("After final_pcd")
    alpha=0.002
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(final_pcd)
    final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=80))
    alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            final_pcd, alpha, tetra_mesh, pt_map)
    pcd = alpha_mesh.sample_points_poisson_disk(3000)
    print("Printing alpha mesh")
    o3d.visualization.draw_geometries([alpha_mesh])
    print("Mesh = ", alpha_mesh.vertices)
    mesh_with_normals = o3d.geometry.TriangleMesh.compute_triangle_normals (alpha_mesh)
    o3d.io.write_triangle_mesh("/home/node1/Output_STL_Files/Part_0/object_model_diff_without_icp.stl", mesh_with_normals)
    
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    ###################################################################

    ##(1) Ball Pivoting 

    print("Starting mesh reconstruction")
    points = np.asarray(final_pcd.points)
    point_cloud = pv.PolyData(points)
    mesh = point_cloud.reconstruct_surface()
    mesh.plot()
    mesh.save('/home/catkin_ws/src/node1/Output_STL_Files/Part_0/pv_object_model_without_icp.stl')
    
    tri = np.asarray(o3d.bpa_mesh.triangles)
    faces = np.empty((tri.shape[0], 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = tri
    mesh = pv.PolyData(final_pcd.points, faces)
    print("Creating mesh")
    print("Mesh = ", mesh)
    mesh.save('/home/catkin_ws/src/node1/Output_STL_Files/Part_0/pv_object_model_without_icp.stl')
    mesh.plot()

    factor = 1.0
    radius = factor * avg_dist   
    radi =  o3d.utility.DoubleVector([radius, radius * 2])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radi)
    o3d.visualization.draw_geometries([mesh])
    mesh_with_normals = o3d.geometry.TriangleMesh.compute_triangle_normals (mesh)
    o3d.io.write_triangle_mesh("/home/catkin_ws/src/node1/gripper_model_bpa_1.stl", mesh_with_normals)

    factor = 5.0
    radius = factor * avg_dist
    radi =  o3d.utility.DoubleVector([radius, radius * 2])
    mesh_2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radi)
    o3d.visualization.draw_geometries([mesh_2])

    
    radii = [0.005, 0.008, 0.01, 0.012]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
    #print("Rec_mesh = ", np.asarray(rec_mesh.vertices))
    o3d.visualization.draw_geometries([rec_mesh])


    #########################################################
    #Dbscan clustering
    #########################################################

    labels = np.array(final_pcd.cluster_dbscan(eps=0.01, min_points=10))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # Visualize segmented point cloud
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    final_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #Take only the object points
    object_indices = np.where(labels != -1)[0]
    object_pcd = final_pcd.select_by_index(object_indices)
    cropped_pc = create_bounding_box(object_pcd)
    o3d.visualization.draw_geometries([cropped_pc])

    #(1)Poisson reconstruction(Not great)
    # #Take only the object points
    cropped_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cropped_pc, depth=5)
        bbox = cropped_pc.get_axis_aligned_bounding_box()
        p_mesh_crop = mesh.crop(bbox)
    o3d.visualization.draw_geometries([p_mesh_crop])

    
    #############################################
    #Working code for clustering
    #############################################

    #Filter outliers
    #final_pcd = filter_outliers(merged_point_cloud)
    #o3d.visualization.draw_geometries([final_pcd])
    
    #final_pcd = merged_point_cloud


    #########################################################
    #Spectral clustering
    #########################################################
    # points = np.asarray(final_pcd.points)
    # clustering = SpectralClustering(n_clusters=5,
    # assign_labels='kmeans',
    # random_state=0).fit(points) 
    # print(f"Clustering labels = ", clustering.labels_)
    # labels = clustering.labels_
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")

    # # Visualize segmented point cloud
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # final_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # #Take only the object points
    # object_indices = np.where(labels == 2)[0]
    # print("Object indices = ", object_indices)
    # object_pcd = final_pcd.select_by_index(object_indices)
    # print("Object pcd = ", object_pcd)
    # cropped_pc = create_bounding_box(object_pcd)


    #########################################################
    #Kmeans clustering
    #########################################################
    # Trying sklearn
    # scaled_points = StandardScaler().fit_transform(np.asarray(final_pcd.points)) 
    # model = KMeans(n_clusters=7)
    # model.fit(scaled_points)
            
    # # Get labels:
    # labels = model.labels_
    # # Get the number of colors:
    # n_clusters = len(set(labels))

    # # Mapping the labels classes to a color map:
    # colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
    # # Attribute to noise the black color:
    # colors[labels < 0] = 0
    # # Update points colors:
    # final_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # min_bound = np.array([-.2,-.2,-1])*.8
    # max_bound = np.array([.2,.2,1])*.8
    # # min_bound = np.array([0.2,0.2,1])*.4
    # # max_bound = np.array([.1,.1,1])*.4
    # vol = o3d.visualization.SelectionPolygonVolume()
    # vol.orthogonal_axis = "Y"
    # vol.axis_max = 0.2
    # vol.axis_min = -0.2
    # bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # pcd_crop = vol.crop_point_cloud(final_pcd)
    #     # Display:
    # o3d.visualization.draw_geometries([pcd_crop])

    

    #(2) Triangular mesh 
    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(cropped_pc)
    # for alpha in np.logspace(np.log10(0.18), np.log10(0.008), num=7):
    #     print(f"alpha={alpha:.3f}")
    #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    #         cropped_pc, alpha, tetra_mesh, pt_map)
    #     mesh.compute_vertex_normals()
    # #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # ##(3) Ball pivoting
    # cropped_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    #         cropped_pc, alpha, tetra_mesh, pt_map)
    # pcd = mesh.sample_points_poisson_disk(3000)
    # #o3d.visualization.draw_geometries([pcd])
    # print("Mesh = ", mesh.vertices)

    # radii = [0.005, 0.01, 0.02, 0.04]
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    # pcd, o3d.utility.DoubleVector(radii))
    # print("Rec_mesh = ", np.asarray(rec_mesh.vertices))
    #o3d.visualization.draw_geometries([pcd, rec_mesh])

    # #########################################################
    # # Cropping object from gripper
    # #########################################################
    # #o3d.visualization.draw_geometries([merged_point_cloud])
    # mesh = o3d.io.read_triangle_mesh("/home/optonic-shared-pc/catkin_ws/src/node1/Finger_15cm.stl")
    # gripper_pointcloud = mesh.sample_points_poisson_disk(100000)
    

    # #Make the transformation for gripper so that the gripper is aligned with thc current point cloud 
    # transformed_gripper = align_point_cloud_gripper(gripper_pointcloud)
    # o3d.visualization.draw_geometries([transformed_gripper, merged_point_cloud])

    # #Calculate the difference in point cloud to get the object
    # dists = merged_point_cloud.compute_point_cloud_distance(transformed_pc)   
    # dists = np.asarray(dists)
    # ind = np.where(dists > 0.01)[0]
    # print("ind = ", ind)
    # pcd_finger = merged_point_cloud.select_by_index(ind, invert= True)
    # o3d.visualization.draw_geometries([pcd_finger])
    # #print("PCD without finger = ", pcd_without_finger.points)
    # #o3d.visualization.draw_geometries([merged_point_cloud])
    # #o3d.visualization.draw_geometries([pcd_without_finger])
    
    ##Subtract the gripper points from the point cloud
    #pcd_without_finger = pcd.points - transformed_gripper.points
    
    # Calculate k nearest neighbours for the gripper point cloud
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # print("PCD tree = ", pcd_tree)
    # o3d.visualization.draw_geometries([pcd_tree])

    