from flask import Flask, request, jsonify
from tasks import generate_video
from celery.result import AsyncResult
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)

@app.route('/')
def home():
    return "<h2>SadTalker API</h2><p>Upload an image and audio to generate a talking face video.</p>"

@app.route('/generate-video', methods=['POST'])
def generate_video_endpoint():
    # Handle image and audio upload
    source_image = request.files.get('source_image')
    driven_audio = request.files.get('driven_audio')

    if not source_image or not driven_audio:
        return jsonify({"error": "Both image and audio are required."}), 400

    # Save the uploaded files temporarily
    image_path = os.path.join('uploads', source_image.filename)
    audio_path = os.path.join('uploads', driven_audio.filename)

    source_image.save(image_path)
    driven_audio.save(audio_path)

    # Prepare arguments for the task
    args = {
        'source_image': image_path,
        'driven_audio': audio_path,
        'checkpoint_dir': './checkpoints',  # Adjust based on your structure
        'result_dir': './results',  # Adjust based on your structure
        'pose_style': 0,  # Default value, can modify based on your needs
        'batch_size': 2,  # Default value
        'size': 256,  # Default value
        'expression_scale': 1.0,  # Default value
        'input_yaw': None,  # Modify as needed
        'input_pitch': None,  # Modify as needed
        'input_roll': None,  # Modify as needed
        'ref_eyeblink': None,  # Modify as needed
        'ref_pose': None,  # Modify as needed
        'still': False,  # Default value
        'face3dvis': False,  # Default value
        'enhancer': None,  # Modify as needed
        'background_enhancer': None,  # Modify as needed
        'verbose': False,  # Default value
        'old_version': False,  # Default value
        'preprocess': 'crop'  # Default value, adjust as necessary
    }

    # Call the Celery task to generate the video
    task = generate_video.apply_async(args=[args])

    return jsonify({"task_id": task.id}), 202  # Return the task ID for tracking

@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task_result = AsyncResult(task_id)

    if task_result.state == 'PENDING':
        # Task has not yet run
        response = {
            'state': task_result.state,
            'status': 'Pending...'
        }
    elif task_result.state != 'FAILURE':
        # Task has finished successfully
        response = {
            'state': task_result.state,
            'video_path': task_result.result  # Assuming result contains the video path
        }
    else:
        # Task failed
        response = {
            'state': task_result.state,
            'error': str(task_result.info)  # This could be the error message from the task
        }

    return jsonify(response)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Ensure the upload directory exists
    app.run(host='0.0.0.0', port=5000)
