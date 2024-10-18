from celery import Celery
from flask import Flask, request, jsonify
from tasks import generate_video
from celery.result import AsyncResult
import os


app = Flask(__name__)

# Configure Celery with Redis as the broker
#REDIS_URI = 'redis://default:redispw@localhost:32768/0'
REDIS_URI = 'redis://localhost:6379/0'
app.config['CELERY_BROKER_URL'] = REDIS_URI
app.config['CELERY_RESULT_BACKEND'] = REDIS_URI


from celery import Celery, Task

def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app

app.config.from_mapping(
    CELERY=dict(
        broker_url=REDIS_URI,
        result_backend=REDIS_URI,
        task_ignore_result=True,
    ),
)
celery_app = celery_init_app(app)

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
        'pose_style': 0,  # Default value, can modify based on your needs
        'batch_size': 2,  # Default value
        'expression_scale': 1.0,  # Default value
        'preprocess': 'crop'  # Default value, adjust as necessary
    }

    # Call the Celery task to generate the video
    task = generate_video.apply_async(kwargs=args)

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
    app.run(host='0.0.0.0', port=5000, debug=True)
