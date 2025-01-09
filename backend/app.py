from flask import Flask, request, send_file
from flask_cors import CORS
from gtts import gTTS
import os
import tempfile
import threading
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# Create a temporary directory for audio files
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Cleanup old files periodically
def cleanup_old_files():
    while True:
        current_time = datetime.now()
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if (current_time - file_creation_time).seconds > 300:  # 5 minutes
                try:
                    os.remove(file_path)
                except:
                    pass
        time.sleep(300)  # Run cleanup every 5 minutes

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

@app.route('/speak', methods=['POST'])
def speak():
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')

        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(TEMP_DIR, filename)

        # Generate speech
        tts = gTTS(text=text, lang=language)
        tts.save(filepath)

        return send_file(
            filepath,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)