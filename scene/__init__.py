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

from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from scene.dataset_readers import init_scene, convert2Camera
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, resolution_scales=[1.0], load_coarse=False):
        self.model_path = args.model_path
        self.gaussians = gaussians

        # 初始化scene_info
        scene_info = init_scene(args.source_path, args.n_points)
        self.n_frame = scene_info.n_frame
        self.meas = scene_info.meas.permute(2, 0, 1).cuda()
        self.gt_images = scene_info.gt_images
        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # 初始化Camera
        self.cameras = convert2Camera(scene_info.cam_infos)
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def getCameras(self, scale=1.0):
        return self.cameras