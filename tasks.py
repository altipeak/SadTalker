from celery import Celery
from glob import glob
import shutil
import torch
from time import strftime
import os
import sys
from argparse import Namespace

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# defaults.py

DEFAULT_ARGS = {
    "driven_audio": './examples/driven_audio/bus_chinese.wav',
    "source_image": './examples/source_image/full_body_1.png',
    "ref_eyeblink": None,
    "ref_pose": None,
    "checkpoint_dir": './checkpoints',
    "result_dir": './results',
    "pose_style": 0,
    "batch_size": 2,
    "size": 256,
    "expression_scale": 1.0,
    "input_yaw": None,
    "input_pitch": None,
    "input_roll": None,
    "enhancer": None,
    "background_enhancer": None,
    "cpu": False,
    "face3dvis": False,
    "still": False,
    "preprocess": 'crop',
    "verbose": False,
    "old_version": False,
    "net_recon": 'resnet50',
    "init_path": None,
    "use_last_fc": False,
    "bfm_folder": './checkpoints/BFM_Fitting/',
    "bfm_model": 'BFM_model_front.mat',
    "focal": 1015.0,
    "center": 112.0,
    "camera_d": 10.0,
    "z_near": 5.0,
    "z_far": 15.0,
}


@celery_app.task
def generate_video(args_dict):
    args_dict = {**DEFAULT_ARGS, **args_dict}
    args = Namespace(**args_dict)

    # Set the device based on availability
    args.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Log the parameters
    logger.info("Starting video generation with parameters:")
    logger.info(f"Device: {args.device}")
    logger.info(f"Driven Audio: {args.driven_audio}")
    logger.info(f"Source Image: {args.source_image}")
    logger.info(f"Checkpoint Directory: {args.checkpoint_dir}")
    logger.info(f"Result Directory: {args.result_dir}")
    logger.info(f"Pose Style: {args.pose_style}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Image Size: {args.size}")
    logger.info(f"Expression Scale: {args.expression_scale}")

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    current_root_path = os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'),
                                args.size, args.old_version, args.preprocess)

    # Initialize models
    preprocess_model = CropAndExtract(sadtalker_paths, args.device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, args.device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, args.device)

    # Crop image and extract 3DMM from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    logger.info('3DMM Extraction for source image')

    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size
    )

    if first_coeff_path is None:
        logger.error("Can't get the coeffs of the input")
        return

    # Reference video processing
    ref_eyeblink_coeff_path = process_reference_video(args.ref_eyeblink, preprocess_model, save_dir, args)
    ref_pose_coeff_path = process_reference_video(args.ref_pose, preprocess_model, save_dir,
                                                  args) if args.ref_pose else None

    # Audio to coefficients
    batch = get_data(first_coeff_path, audio_path, args.device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, args.pose_style, ref_pose_coeff_path)

    # 3D face render and coefficient to video
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, args.device, first_coeff_path, coeff_path, audio_path,
                           os.path.join(save_dir, '3dface.mp4'))

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               args.batch_size, args.input_yaw, args.input_pitch, args.input_roll,
                               expression_scale=args.expression_scale, still_mode=args.still,
                               preprocess=args.preprocess, size=args.size)

    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                         enhancer=args.enhancer, background_enhancer=args.background_enhancer,
                                         preprocess=args.preprocess, img_size=args.size)

    final_output_path = f"{save_dir}.mp4"
    shutil.move(result, final_output_path)
    logger.info('The generated video is named: %s', final_output_path)

    if not args.verbose:
        shutil.rmtree(save_dir)


def process_reference_video(ref_video, preprocess_model, save_dir, args):
    if ref_video is not None:
        ref_video_name = os.path.splitext(os.path.split(ref_video)[-1])[0]
        ref_frame_dir = os.path.join(save_dir, ref_video_name)
        os.makedirs(ref_frame_dir, exist_ok=True)
        logger.info(f'3DMM Extraction for the reference video: {ref_video}')
        ref_coeff_path, _, _ = preprocess_model.generate(ref_video, ref_frame_dir, args.preprocess,
                                                         source_image_flag=False)
        return ref_coeff_path
    return None
