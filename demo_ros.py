""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse

import torch
from graspnetAPI import GraspGroup

import rospy
import sensor_msgs
from cv_bridge import CvBridge
import scipy
import geometry_msgs.msg

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default="logs/log_kn/checkpoint-rs.tar", required=False, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

COLOR_IMAGE=None
DEPTH_IMAGE=None
MASK_IMAGE=None
GRASP_POSE_PUB = rospy.Publisher('/grasp_pose', geometry_msgs.msg.Pose, queue_size=1)

def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def dilated_outer_rectangle(foreground_mask):
    # 找到True值的索引范围
    nonzero_indices = np.where(foreground_mask)

    # 计算外接矩形的左上角坐标和宽度、高度
    min_row = np.min(nonzero_indices[0]) - 200
    max_row = np.max(nonzero_indices[0]) + 200
    min_col = np.min(nonzero_indices[1]) - 200
    max_col = np.max(nonzero_indices[1]) + 200

    dilated_outer_rectangle_mask = np.zeros(foreground_mask.shape, dtype=bool)
    dilated_outer_rectangle_mask[min_row:max_row+1, min_col:max_col+1] = True
    return dilated_outer_rectangle_mask

def get_and_process_data():
    # load data
    color = COLOR_IMAGE # 1280 x 720 x 3 , 0.0 ~ 1.0
    depth = DEPTH_IMAGE # 1280 x 720, 0 ~ 1590
    workspace_mask = dilated_outer_rectangle(MASK_IMAGE) # 1280 x 720, True or False

    # 获取相机内参：rostopic echo /camera/color/camera_info
    #  width, height, fx, fy, cx, cy, scale
    camera = CameraInfo(1280.0, 720.0, 911.272, 911.4296, 647.6853, 366.2829, 1000.0)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def project_3d_to_2d(x, y, z, fx, fy, cx, cy):
    u = int(fx * x / z + cx)
    v = int(fy * y / z + cy)
    return u, v

def base_link_transform():
    # 绕x轴转135度
    angle = np.deg2rad(135)
    c = np.cos(angle)
    s = np.sin(angle)
    T1 = np.eye(4)
    T1[1:3, 1:3] = np.array([[c, -s], [s, c]])
    # 绕z轴转180度
    angle = np.deg2rad(180)
    c = np.cos(angle)
    s = np.sin(angle)
    T2 = np.eye(4)
    T2[0:2, 0:2] = np.array([[c, -s], [s, c]])
    # z轴平移-0.67
    T3 = np.eye(4)
    T3[2, 3] = -0.67
    # y轴平移-0.04
    T4 = np.eye(4)
    T4[1, 3] = -0.04
    return T1 @ T2 @ T3 @ T4

def demo(event):
    if COLOR_IMAGE is None or DEPTH_IMAGE is None or MASK_IMAGE is None:
        print('Images are not ready')
        return
    net = get_net()
    end_points, cloud = get_and_process_data()
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    gg.nms()
    gg.sort_by_score()
    gg = gg[:30]
    ######### 排除掉mask外的抓取点 ############
    remove_index = []
    for i in range(len(gg)):
        grasp = gg[i]
        x, y, z = grasp.translation
        u, v = project_3d_to_2d(x, y, z, 911.272, 911.4296, 647.6853, 366.2829)
        if not MASK_IMAGE[v, u]:
            remove_index.append(i)
    for i in remove_index[::-1]:
        gg.remove(i)
    #########################################
    ####### 只保留 TOP1 的抓取点 #############
    for i in range(len(gg)-1, 0, -1):
        gg.remove(i)
    #########################################
    ############# base link坐标系 #############
    T = base_link_transform()
    base_link = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1) # 坐标轴
    base_link = base_link.transform(T)
    o3d.visualization.draw_geometries([cloud
                                       , *gg.to_open3d_geometry_list()
                                       , o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
                                       , base_link
                                       ])

    print("origin tranlation of the best grasp is ", gg[0].translation)
    origin_grasp_transform = np.eye(4)
    origin_grasp_transform[0:3, 0:3] = gg[0].rotation_matrix
    origin_grasp_transform[0:3, 3] = gg[0].translation
    new_grasp_transform = np.linalg.inv(T) @ origin_grasp_transform
    print("new tranlation of the best grasp is ", new_grasp_transform[0:3, 3])
    rotation_obj = scipy.spatial.transform.Rotation.from_matrix(new_grasp_transform[0:3, 0:3])
    quaternion = rotation_obj.as_quat()
    print("new quaternion of the best grasp is ", quaternion)
    ##########################################
    grasp_pose = geometry_msgs.msg.Pose()
    grasp_pose.position.x = new_grasp_transform[0, 3]
    grasp_pose.position.y = new_grasp_transform[1, 3]
    grasp_pose.position.z = new_grasp_transform[2, 3]
    grasp_pose.orientation.x = quaternion[0]
    grasp_pose.orientation.y = quaternion[1]
    grasp_pose.orientation.z = quaternion[2]
    grasp_pose.orientation.w = quaternion[3]
    GRASP_POSE_PUB.publish(grasp_pose)
    return


def color_image_callback(msg):
    # 创建CvBridge对象
    bridge = CvBridge()
    # 将ROS图像消息转换为OpenCV图像格式
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    # 将OpenCV图像转换为NumPy数组
    global COLOR_IMAGE
    COLOR_IMAGE = np.array(cv_image) / 255.0 # rgb的通道顺序, 0.0-1.0, float64
    # 在这里对NumPy数组进行处理
    return

def depth_image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    global DEPTH_IMAGE
    DEPTH_IMAGE = np.array(cv_image)
    return
    
def mask_image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    global MASK_IMAGE
    MASK_IMAGE = np.array(cv_image, dtype=bool)
    return

if __name__ == '__main__':
    rospy.init_node('graspnet_ros_demo')

    # 创建图像订阅者
    _ = rospy.Subscriber('/camera/color/image_raw', sensor_msgs.msg.Image, color_image_callback)
    _ = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', sensor_msgs.msg.Image, depth_image_callback)
    _ = rospy.Subscriber('/detic/target_mask', sensor_msgs.msg.Image, mask_image_callback)
    _ = rospy.Timer(rospy.Duration(1.0), demo)

    # 进入ROS循环
    rospy.spin()