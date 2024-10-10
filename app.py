from flask import Flask, request, jsonify
from tasks import generate_video_task
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)

@app.route('/')
def home():
    return "<h2>SadTalker API</h2><p>Upload an image and audio to generate a talking face video.</p>"

@app.route('/generate-video', methods=['POST'])
def generate_video():
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

    # Call the Celery task to generate the video
    task = generate_video_task.apply_async(args=[image_path, audio_path])

    return jsonify({"task_id": task.id}), 202  # Return the task ID for tracking

from celery.result import AsyncResult

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
            'video_path': task_result.result
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
