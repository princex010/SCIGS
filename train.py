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
import numpy as np
import random
import os, sys
os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
from utils.loss_utils import l1_loss, psnr
from pytorch_msssim import ssim as pssim
import lpips
from gaussian_renderer import render
from scene import Scene, GaussianModel
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.timer import Timer
from scene.deform_model import DeformModel
from PIL import Image

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(dataset, opt, pipe, saving_iterations,
                         checkpoint_iterations, gs_checkpoint, deform_checkpoint, debug_from,
                         gaussians, scene, deform, train_iter, timer):
    lpips_model = lpips.LPIPS(net="alex").cuda()
    lpips_frame = 0.0
    first_iter = 0
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    gaussians.training_setup(opt)
    deform.train_setting(opt)
    if gs_checkpoint and deform_checkpoint:
        (model_params, first_iter) = torch.load(gs_checkpoint)
        gaussians.restore(model_params, opt)
        deform = torch.load(deform_checkpoint)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    viewpoint_cams = scene.getCameras()

    # gt_images = scene.gt_images
    # gt_image_tensor = torch.cat(gt_images, 0).cuda()
    # gt_image_tensor = gt_image_tensor.type(torch.cuda.FloatTensor)
    d_xyzs = [0.0 for i in range(scene.n_frame)]
    gaussians.compute_3D_filter(d_xyzs, cameras=viewpoint_cams)
    print("data loading done")
    for iteration in range(first_iter, final_iter+1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        deform.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        masks = []
        radii_list = []
        d_xyzs = []
        d_rotations = []

        point_num = gaussians.get_xyz.shape[0]

        visibility_filter_all = torch.zeros(point_num, dtype=torch.bool, device=device)

        for viewpoint_cam in viewpoint_cams:
            N = gaussians.get_xyz.shape[0]
            time = torch.tensor(viewpoint_cam.time)
            time_input = time.unsqueeze(0).expand(N, -1).cuda()
            d_xyz, d_rotation, _ = deform.step(gaussians.get_xyz.detach(), time_input)

            d_xyzs.append(d_xyz)
            d_rotations.append(d_rotation)


        for idx, viewpoint_cam in enumerate(viewpoint_cams):

            render_pkg = render(viewpoint_cam, gaussians, pipe, background, d_xyzs[idx], d_rotations[idx])
            image, viewspace_point_tensor, viewspace_point_tensor_densify, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["viewspace_points_densify"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            mask = viewpoint_cam.mask.cuda()
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_all = torch.logical_or(visibility_filter, visibility_filter_all)
            if iteration < args.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])

        radii = torch.cat(radii_list,0).max(dim=0).values
        image_tensor = torch.cat(images, 0)
        mask_tensor = torch.cat(masks, 0)
        masked_images = image_tensor * mask_tensor
        sci_image = masked_images.sum(dim=0)
        # Loss
        # breakpoint()
        Ll1 = l1_loss(sci_image, scene.meas)
        L_ssim = pssim(sci_image.unsqueeze(0) * 255, scene.meas.unsqueeze(0) * 255)

        # Ll1_frame = l1_loss(image_tensor, gt_image_tensor).mean().double()

        # psnr_frame = psnr(image_tensor, gt_image_tensor).mean().double()

        # ssim_frame = pssim(image_tensor * 255, gt_image_tensor * 255)

        # if iteration % 200 == 0:
        #     lpips_frame = lpips_model(image_tensor, gt_image_tensor).mean().double()

        loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1 - L_ssim)
        # loss = Ll1

        
        loss.backward()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}",
                                        #   "psnr_f": f"{psnr_frame:.{3}f}",
                                        #   "ssim_f": f"{ssim_frame:.{4}f}",
                                        #   "lpips_f":f"{lpips_frame:.{4}f}", 
                                        #   "l1": f"{Ll1:.{7}f}",
                                        #   "psnr": f"{psnr_sci:.{3}f}",                                         
                                          "point": f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                gaussians.save(iteration, dataset.model_path)
                deform.save_weights(args.model_path, iteration)
                if iteration in checkpoint_iterations:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    torch.save(deform, scene.model_path + "/deform_chkpnt" + str(iteration) + ".pt")
                i = 0
                for tensor in images:
                    image_save = tensor.squeeze(0).permute(1, 2, 0)
                    image_np = image_save.clone().detach().cpu().numpy()
                    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                    image_pil.save(os.path.join(dataset.images, "image" + str(iteration) + "cam" + str(i) + ".png"))
                    i += 1

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter_all)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    new_dxyzs = gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, d_xyzs)
                    if iteration >= 2000:
                        print(0)
                    gaussians.compute_3D_filter(new_dxyzs, cameras=viewpoint_cams)                 

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(d_xyzs, cameras=viewpoint_cams)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.step()
                deform.update_learning_rate(iteration)              
                deform.optimizer.zero_grad(set_to_none=True)
                




def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, gs_checkpoint, deform_checkpoint, debug_from, expname):
    # first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset, expname)
    gaussians = GaussianModel(dataset.sh_degree)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    deform = DeformModel()
    timer.start()
    scene_reconstruction(dataset, opt, pipe, saving_iterations,
                         checkpoint_iterations, gs_checkpoint, deform_checkpoint, debug_from,
                         gaussians, scene, deform, opt.iterations, timer)

def prepare_output_and_logger(args, expname):
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 4000, 7000, 14000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--gs_checkpoint", type=str, default = None)
    parser.add_argument("--deform_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.gs_checkpoint, args.deform_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
