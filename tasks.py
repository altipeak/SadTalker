from celery import Celery
from src.gradio_demo import SadTalker
import os

# Initialize Celery
app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task(bind=True)
def generate_video_task(self, image_path, audio_path):
    sad_talker = SadTalker(checkpoint_path='checkpoints', config_path='src/config', lazy_load=True)

    # Generate video
    video_path = sad_talker.test(source_image=image_path, driven_audio=audio_path)

    # Return the path to the generated video
    return video_path
