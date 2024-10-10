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

celery_app = Celery('tasks', broker='redis://localhost:6379/0')


@celery_app.task
def generate_video(args_dict):
    args = Namespace(**args_dict)

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    current_root_path = os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size,
                                args.old_version, args.preprocess)

    # Initialize models
    preprocess_model = CropAndExtract(sadtalker_paths, args.device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, args.device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, args.device)

    # Crop image and extract 3DMM from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')

    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,
                                                                           source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    # Reference video processing
    ref_eyeblink_coeff_path = process_reference_video(args.ref_eyeblink, preprocess_model, save_dir, args)
    ref_pose_coeff_path = process_reference_video(args.ref_pose, preprocess_model, save_dir,
                                                  args) if args.ref_pose else None

    # Audio to coefficient
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
    print('The generated video is named:', final_output_path)

    if not args.verbose:
        shutil.rmtree(save_dir)


def process_reference_video(ref_video, preprocess_model, save_dir, args):
    if ref_video is not None:
        ref_video_name = os.path.splitext(os.path.split(ref_video)[-1])[0]
        ref_frame_dir = os.path.join(save_dir, ref_video_name)
        os.makedirs(ref_frame_dir, exist_ok=True)
        print(f'3DMM Extraction for the reference video: {ref_video}')
        ref_coeff_path, _, _ = preprocess_model.generate(ref_video, ref_frame_dir, args.preprocess,
                                                         source_image_flag=False)
        return ref_coeff_path
    return None
