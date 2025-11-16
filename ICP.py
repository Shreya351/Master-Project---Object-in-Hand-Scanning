import open3d as o3d
import os
import re
import rospy
import numpy as np
import copy
import time
import pyvista as pv
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointField

#import mikado_shreya as mk


def read_file(file_name):
    file_handle = open(file_name)
    pose = file_handle.read()
    print("pose = ", pose)
    file_handle.close()
    return pose


def getpose(i, path):
    # Read the text file
    # Convert to geometry_msg/Pose.msg
    #
    # Append it to pose list

    file_dir = os.path.realpath(path)
    #/home/optonic-shared-pc/Documents/Work/Mikado_FileCam/Part_1/')#'/home/optonic-shared-pc/Documents/Work/FileCamera_New/')
    #/home/optonic-shared-pc/Documents/Work/FileCamera/Work/')
    print(file_dir)
    # Remember to change str(i+1) to str(i) for new dataset
    file_path = 'pose' + str(i) + '.txt'
    #print("file_path = ", file_path)
    file_name = os.path.join(file_dir, file_path)
    pose = read_file(file_name)
    with open('/home/optonic-shared-pc/Documents/Work/Mikado_FileCam/Part_1/All.txt', 'a') as f:
        f.write(str(i))
        f.write('\n')
        f.write(pose)
        f.write('\n')
    return pose

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


def get_transform(pose):
    xyz = separate_xyz_fromstring(pose)[0]
    rpy = separate_xyz_fromstring(pose)[1]
    Rot_mat = Rotation.from_euler('XYZ', rpy, degrees=False).as_matrix()
    Trans_mat = xyz

    Transformed_matrix = np.empty((4, 4))
    Transformed_matrix[:3, :3] = Rot_mat
    Transformed_matrix[:3, 3] = Trans_mat
    Transformed_matrix[3, :] = [0, 0, 0, 1]
    Tinv = np.linalg.inv(Transformed_matrix)
    Tinv = np.linalg.inv(Transformed_matrix)
    return Tinv

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    final_pcd = source_temp.transform(transformation)
    #o3d.visualization.draw_geometries([source_temp, target_temp])
    return final_pcd

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
        # (why cannot put this line below rgb?)
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


def icp(source, target, trans_init):
    threshold = 0.02
    print("Apply point-to-point ICP")
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)     

    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(source, target, reg_p2l.transformation)
    return reg_p2l.transformation

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.uniform_down_sample(8)
    print("pcd_down: ", pcd_down)
    radius_normal = voxel_size * 0.002
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 0.002
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset_for_icp(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("/home/optonic-shared-pc/catkin_ws/src/node1/Output_STL_Files/Part_0/object_pcd.pcd")
    target = o3d.io.read_point_cloud(
             "/home/optonic-shared-pc/catkin_ws/src/node1/Output_STL_Files/Part_0_differentGrasp/object_pcd.pcd")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.008
    print(":: 2 Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def apply_transformation(source, target):
    Rot_mat = Rotation.from_euler('XYZ', [3.2, 1.2, 2.2], degrees=False).as_matrix()
    Trans_mat = np.array([0.7, 0.1, 0.2])

    Transformed_matrix = np.empty((4, 4))
    Transformed_matrix[:3, :3] = Rot_mat
    Transformed_matrix[:3, 3] = Trans_mat
    Transformed_matrix[3, :] = [0, 0, 0, 1]
    
    Tinv = np.linalg.inv(Transformed_matrix)
    #pcd = convertCloudFromRosToOpen3d(source)
    aligned_pc = source.transform(Tinv)
    o3d.visualization.draw_geometries([source, target])
    return aligned_pc

if __name__ == '__main__':

    voxel_size = 0.002
    start = time.time()
    
    source,target,source_down,target_down,source_fpfh,target_fpfh = prepare_dataset_for_icp(voxel_size)
    o3d.visualization.draw_geometries([source])
    o3d.visualization.draw_geometries([target])
    source = apply_transformation(source, target)
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print("result_fast =  ", result_fast)
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=80))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=80))
    final_transform = icp(source_down, target_down, result_fast.transformation)
    
    final_pcd = source_down.transform(final_transform)
    o3d.visualization.draw_geometries([final_pcd])
    print("########## PCD_down ########### = ", np.asarray(final_pcd.points))

    ##Ball pivoting algorithm
    alpha = 0.0025
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(final_pcd)
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                final_pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([final_pcd, rec_mesh])
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            final_pcd, alpha, tetra_mesh, pt_map)
    pcd = mesh.sample_points_poisson_disk(3000)
    
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("/home/optonic-shared-pc/catkin_ws/src/node1/Output_STL_Files/Part_0/Final_object_model_ballPivot.stl", mesh)


            


    #########################################
    ## Mesh reconstruction with PyVista##
    #########################################
    # pcd = pv.PolyData(np.asarray(final_pcd.points))
    # print("PV pcd  = ", pcd)
    # mesh = pcd.reconstruct_surface()
    # mesh.plot(color='white')
    # mesh.save('/home/optonic-shared-pc/catkin_ws/src/node1/Output_STL_Files/Part_0/pv_object_model.stl')
    
    
    
    ########### Mesh Reconstruction #####s.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=80))######
    # final_pcd = final_pcd.random_down_sample(1.0)
    # print("final_pcd = ", final_pcd.points)
    # alpha=0.0031
    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(final_pcd)
    # final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=80))
    # alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    #         final_pcd, alpha, tetra_mesh, pt_map)
    # pcd = alpha_mesh.sample_points_poisson_disk(3000)
    # print("Printing alpha mesh")
    # o3d.visualization.draw_geometries([alpha_mesh])
    # print("Mesh = ", alpha_mesh.vertices)
    # mesh_with_normals = o3d.geometry.TriangleMesh.compute_triangle_normals (alpha_mesh)
    # o3d.io.write_triangle_mesh("/home/optonic-shared-pc/catkin_ws/src/node1/Output_STL_Files/Part_0/Final_object_model.stl", mesh_with_normals)
    
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)

    # ##Poisson reconstruction(Not great)

    # with o3d.utility.VerbosityContextManager(
    #     o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #     final_pcd, depth=5)
    #     bbox = final_pcd.get_axis_aligned_bounding_box()
    #     p_mesh_crop = mesh.crop(bbox)
    # p_mesh_crop.compute_vertex_normals()
    # #o3d.visualization.draw_geometries([p_mesh_crop])
    # o3d.io.write_triangle_mesh("/home/optonic-shared-pc/catkin_ws/src/node1/Output_STL_Files/Part_0/Final_object_model_poisson.stl", p_mesh_crop)

    # ##Ball pivoting algorithm
    # alpha = 0.0025
    # radii = [0.005, 0.01, 0.02, 0.04]
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #             final_pcd, o3d.utility.DoubleVector(radii))
    #o3d.visualization.draw_geometries([final_pcd, rec_mesh])
    
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    #         final_pcd, alpha, tetra_mesh, pt_map)
    # pcd = mesh.sample_points_poisson_disk(3000)
    
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    # o3d.io.write_triangle_mesh("/home/optonic-shared-pc/catkin_ws/src/node1/Output_STL_Files/Part_0/Final_object_model_ballPivot.stl", mesh)

        

