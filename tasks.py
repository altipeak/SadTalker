import subprocess

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

REDIS_URI = 'redis://default:redispw@localhost:32768/0'
#REDIS_URI = 'redis://localhost:6379/0'
celery_app = Celery('tasks', broker=REDIS_URI, backend=REDIS_URI)

@celery_app.task(bind=True)
def generate_video(self, **args_dict):
    """
    Task to generate video by running the inference.py script via subprocess.

    Args:
        args_dict (dict): Dictionary of arguments to pass to inference.py.

    Returns:
        str: Output or error message from the subprocess.
    """
    try:
        args = Namespace(**args_dict)

        # Get the Celery task ID
        task_id = self.request.id

        # Construct the command with arguments
        command = ['python', 'inference.py']

        # Add all arguments to the command
        for key, value in vars(args).items():
            if value is True:
                command.append(f'--{key}')
            if value is not None:  # Skip None values
                command.append(f'--{key}')
                if isinstance(value, list):
                    command.extend(value)  # If the argument is a list, extend the command
                else:
                    command.append(str(value))  # Convert the value to string

        # Add the Celery task ID as an additional argument
        command.append('--task_id')
        command.append(task_id)

        # Run the subprocess command
        logger.info(command)
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Check for successful execution
        if result.returncode == 0:
            logger.info(f"Video generated successfully! Output: {result.stdout}")
        else:
            logger.error(f"Failed to generate video. Error: {result.stderr}")

    except subprocess.CalledProcessError as e:
        return f"Subprocess error: {e.stderr}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"



@celery_app.task
def _generate_video(args_dict):
    #args_dict = {**DEFAULT_ARGS, **args_dict}
    args = Namespace(**args_dict)

    # Set the device based on availability

    if torch.cuda.is_available():
        args.device = "cuda"
    #elif torch.backends.mps.is_available():
    #    args.device = "mps"
    else:
        args.device = "cpu"

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


    current_root_path = os.path.dirname(os.path.abspath(__file__))
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
