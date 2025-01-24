#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from scene.cameras import Camera

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
import math
class CameraInfo(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    FovY: float
    FovX: float
    focal_x: int
    focal_y: int
    width: int
    height: int
    znear: float
    zfar: float
    time : float
    mask: torch.Tensor
   
class SceneInfo(NamedTuple):
    meas: torch.Tensor
    gt_images: list
    point_cloud: BasicPointCloud
    cam_infos: list
    nerf_normalization: dict
    n_frame: int
    ply_path: str
    maxtime: int

def init_cam_infos(H, W, mask, n_frame, znear=1.0, zfar=100.0):
    H, W = int(H), int(W)
    fovx = 1.4731580175054242
    fovy = 0.6174722164096383
    R = torch.tensor([[0.99992703, -0.00161595, 0.01197153], [0.00159289, 0.99999686, 0.00193603], [-0.01197462, -0.00191682, 0.99992646]], dtype=torch.float32)
    t = torch.tensor([5.0587, -0.0179, 0.0508], dtype=torch.float32)
    focal_x = 330.8206182336591
    focal_y = 627.0880288343071

    # H, W = int(H), int(W)
    # fovx = 0.8743119484290753
    # fovy = 0.7851041815496814  
    # R = torch.tensor([[ 0.97867452, 0.01406929, 0.20493472], [ 0.0167664,  0.98885183, -0.14795586], [-0.20473171, 0.14823665, 0.9675282 ]], dtype=torch.float32)
    # t = torch.tensor([ 3.7763,  1.1108, -0.4090], dtype=torch.float32)
    # focal_x = W / (2 * np.tan(fovx / 2))
    # focal_y = H / (2 * np.tan(fovy / 2))
    # poses = torch.zeros((3, 5))
    # poses = poses[np.newaxis, :]
    # poses_se3 = SE3_to_se3_N(poses[:, :3, :4])

    # H, W, focal = hwf
    # H, W = int(H), int(W)

    # low, high = 0.0001, 0.005

    # rand = (high - low) * torch.rand(poses_se3.shape[0], 6) + low
    # if focal < 300:
    #     low_x, high_x = 0.5, 1.0
    #     rand_x = (high_x - low_x) * torch.rand(1) + low_x
    #     rand[0, 3] = -1 * rand_x

    # else:
    #     low_x, high_x = 0.2, 0.5
    #     rand_x = (high_x - low_x) * torch.rand(1) + low_x
    #     rand[0, 3] = -1 * rand_x
    # poses_se3 = rand  # if pose_end is not identical to pose_start, there is no need to add perturb
    # q, t = se3_2_qt_parallel(poses_se3)
    # R = q_to_R_parallel(q)
    # fovx = focal2fov(focal, W)
    # fovy = focal2fov(focal, H)
    times = np.linspace(0.001, 1 - 0.001, n_frame, dtype=np.float32)

    cam_infos = []
    for i in range(n_frame):
        cam_info = CameraInfo(R=R, T=t, FovY=fovy, FovX=fovx, focal_x=focal_x, focal_y=focal_y, width=W, height=H, znear=znear, zfar=zfar, time=times[i], mask=mask[i])
        cam_infos.append(cam_info)

    return cam_infos

def random_initialize(point_num, camera):
    points_image = np.random.uniform(-1, 1, (point_num, 2))
    depths = np.random.random(point_num)
    points_camera = np.zeros([point_num, 3])
    points_camera[:, 2] = camera.znear + depths * (camera.zfar - camera.znear)
    xfar = camera.zfar * math.tan(0.5 * camera.FovX)
    yfar = camera.zfar * math.tan(0.5 * camera.FovY)
    points_camera[:, 0] = points_image[:, 0] * xfar * points_camera[:, 2] / camera.zfar
    points_camera[:, 1] = points_image[:, 1] * yfar * points_camera[:, 2] / camera.zfar
    points_camera = points_camera - camera.T.numpy()
    points_world = np.matmul(camera.R.T.squeeze(-1).numpy(), points_camera.T)
    xyz = points_world.T

    rgb = np.random.rand(xyz.shape[0], 3)
    return xyz, rgb

def convert2Camera(cam_infos):
    Cameras = []
    for cam_info in cam_infos:
        cam = Camera(R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY, focal_x=cam_info.focal_x, focal_y=cam_info.focal_y, mask=cam_info.mask.permute(2, 0, 1), W=cam_info.width, H=cam_info.height,time=cam_info.time)
        Cameras.append(cam)
    return Cameras

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time = float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def init_scene(meas_path, n_point):
    # 导入sci_mea,mask,gt_image
    gt_images = []
    gt_path = os.path.join(meas_path, 'images')
    gt_list = sorted(os.listdir(gt_path))
    for f in gt_list:
        image = Image.open(os.path.join(gt_path, f))
        image_np = np.array(image)
        gt_images.append(torch.tensor(image_np / 255.0).permute(2, 0, 1).unsqueeze(0))
    diffMask = np.load(os.path.join(meas_path, 'mask.npy'))
    meas = np.load(os.path.join(meas_path, 'meas.npy'))
    H, W = meas.shape[0], meas.shape[1]
    diffMask = torch.Tensor(diffMask)
    diffMask = diffMask.unsqueeze(-1)
    meas = torch.Tensor(meas)
    # 初始化相机
    n_frame = diffMask.shape[0]
    caminfos = init_cam_infos(H, W, diffMask, n_frame)
    # 在相机视锥内初始化点云，并将点云存储为ply
    xyz, rgb = random_initialize(n_point, caminfos[0])
    radius = 6.458774852752686
    nerf_normalization = {"translate": np.array([0.0, 0.0, 0.0]), "radius": radius}

    ply_path = os.path.join(meas_path, "pcd/points3D.ply")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)

    except:
        pcd = None
    normals = np.zeros_like(xyz)
    pcd = BasicPointCloud(xyz, rgb, normals)

    scene_info = SceneInfo(meas=meas,
                           gt_images=gt_images,
                           point_cloud=pcd,
                           cam_infos=caminfos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           n_frame=n_frame,
                           ply_path=ply_path)
    return scene_info


