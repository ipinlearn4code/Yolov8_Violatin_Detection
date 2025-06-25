from flask import Flask, Response
import cv2
import numpy as np

app = Flask(__name__)

# Inisialisasi webcam
camera = cv2.VideoCapture(0)

def generate_mjpeg_frames():
    while True:
        # Baca frame dari webcam
        success, frame = camera.read()
        if not success:
            break
        else:
            # Kompresi frame ke format JPEG dengan kualitas tertentu
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame = buffer.tobytes()
            
            # Yield frame dalam format MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + 
                   frame + b'\r\n')

@app.route('/')
def index():
    # Halaman HTML sederhana untuk menampilkan stream MJPEG
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Webcam MJPEG Stream</title>
    </head>
    <body>
        <img src="/video_feed" style="width:640px;height:480px;">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    # Return MJPEG stream response
    
    return Response(generate_mjpeg_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5050, debug=True)
    finally:
        # Bersihkan resource webcam saat aplikasi berhenti
        camera.release()
        cv2.destroyAllWindows()