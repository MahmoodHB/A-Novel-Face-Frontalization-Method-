import os
import io
import base64
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string, request, jsonify
from keras.models import load_model, Model
from skimage import img_as_float
import face_alignment
import multipie_gen

# ===========================
# TP-GAN MODEL SETUP
# ===========================
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

EYE_H, EYE_W = 40, 40
NOSE_H, NOSE_W = 32, 40
MOUTH_H, MOUTH_W = 32, 48

class CloudableModel(Model):
    def load_weights(self, filepath, by_name=False):
        print(f'Loading weights: {filepath}')
        super().load_weights(filepath=filepath, by_name=by_name)

def build_model(model_path):
    model = load_model(model_path, custom_objects={
        'CloudableModel': CloudableModel,
        'tf': tf,
        'multipie_gen': multipie_gen
    })
    return model

def reshape_4d(img):
    return np.expand_dims(img, axis=0)

def reshape_3d(img):
    return img[0]

def img2float(img):
    return img_as_float(img)

# ===========================
# FULL CROP FUNCTION
# ===========================
def crop(img, landmarks, size=128):
    eye_y = 40/size
    mouth_y = 88/size
    
    reye = np.average(np.array((landmarks[37], landmarks[38], landmarks[40], landmarks[41])), axis=0)
    leye = np.average(np.array((landmarks[43], landmarks[44], landmarks[46], landmarks[47])), axis=0)
    mouth = np.average(np.array((landmarks[48], landmarks[54])), axis=0)
    nose_tip = landmarks[30]
    
    vec_mouth2reye = reye - mouth
    vec_mouth2leye = leye - mouth
    phi = np.arccos(vec_mouth2reye.dot(vec_mouth2leye) / 
                    (np.linalg.norm(vec_mouth2reye) * np.linalg.norm(vec_mouth2leye)))/np.pi * 180
    
    if phi < 15:
        eye_center = (reye + leye) / 2
        if nose_tip[0] > eye_center[0]:
            leye = reye
        else:
            reye = leye
    
    if np.array_equal(reye, leye) or phi < 38:
        theta = 0
    else:
        vec_leye2reye = reye - leye
        if vec_leye2reye[0] < 0:
            vec_leye2reye = -vec_leye2reye
        theta = np.arctan(vec_leye2reye[1]/vec_leye2reye[0])/np.pi*180
    
    imgcenter = (img.shape[1]/2, img.shape[0]/2)
    rotmat = cv2.getRotationMatrix2D(imgcenter, theta, 1)
    rot_img = cv2.warpAffine(img, rotmat, (img.shape[1], img.shape[0])) 
    rot_landmarks = np.transpose(rotmat[:, :2].dot(np.transpose(landmarks)) + 
                                 np.repeat(rotmat[:, 2].reshape((2,1)), landmarks.shape[0], axis=1))
    
    rot_reye = np.average(np.array((rot_landmarks[37], rot_landmarks[38], rot_landmarks[40], rot_landmarks[41])), axis=0)
    rot_leye = np.average(np.array((rot_landmarks[43], rot_landmarks[44], rot_landmarks[46], rot_landmarks[47])), axis=0)
    rot_mouth = np.average(np.array((rot_landmarks[48], rot_landmarks[54])), axis=0)
    
    crop_size = int((rot_mouth[1] - rot_reye[1])/(mouth_y - eye_y) + 0.5)
    crop_up = int(rot_reye[1] - crop_size * eye_y + 0.5)
    crop_left = int((rot_reye[0] + rot_leye[0]) / 2 - crop_size / 2 + 0.5)
  
    up_size_fixed=0
    left_size_fixed=0
    
    if crop_up < 0 :
        up_size_fixed=abs(crop_up)
        crop_up = 0
    
    if crop_left < 0 :
        left_size_fixed=abs(crop_left)
        crop_left = 0
            
    crop_img = rot_img[crop_up:crop_up+crop_size+up_size_fixed, crop_left:crop_left+crop_size+left_size_fixed]
    crop_landmarks = rot_landmarks - np.array([crop_left, crop_up])
    
    crop_img = cv2.resize(crop_img, (size, size))
    crop_landmarks *= size / crop_size

    # Extract eyes, nose, mouth
    leye_points = crop_landmarks[42:48]
    leye_center = (np.max(leye_points, axis=0) + np.min(leye_points, axis=0)) / 2
    leye_img = crop_img[int(leye_center[1]-EYE_H/2):int(leye_center[1]+EYE_H/2),
                        int(leye_center[0]-EYE_W/2):int(leye_center[0]+EYE_W/2)]

    reye_points = crop_landmarks[36:42]
    reye_center = (np.max(reye_points, axis=0) + np.min(reye_points, axis=0)) / 2
    reye_img = crop_img[int(reye_center[1]-EYE_H/2):int(reye_center[1]+EYE_H/2),
                        int(reye_center[0]-EYE_W/2):int(reye_center[0]+EYE_W/2)]

    nose_points = crop_landmarks[31:36]
    nose_center = (np.max(nose_points, axis=0) + np.min(nose_points, axis=0)) / 2
    nose_img = crop_img[int(nose_center[1]-NOSE_H/2):int(nose_center[1]+NOSE_H/2),
                        int(nose_center[0]-NOSE_W/2):int(nose_center[0]+NOSE_W/2)]

    mouth_points = crop_landmarks[48:60]
    mouth_center = (np.max(mouth_points, axis=0) + np.min(mouth_points, axis=0)) / 2
    mouth_img = crop_img[int(mouth_center[1]-MOUTH_H/2):int(mouth_center[1]+MOUTH_H/2),
                         int(mouth_center[0]-MOUTH_W/2):int(mouth_center[0]+MOUTH_W/2)]

    return crop_img, leye_img, reye_img, nose_img, mouth_img

# ===========================
# GENERATE FRONTAL FACE
# ===========================
def generate_frontal_face_full(model, side_face_img):
    landmarks = fa.get_landmarks(cv2.cvtColor(side_face_img, cv2.COLOR_BGR2RGB))
    if landmarks is None:
        raise ValueError("No face landmarks detected.")

    crop_im, leye_im, reye_im, nose_im, mouth_im = crop(side_face_img, landmarks[0], size=128)

    inputs = [
        reshape_4d(img2float(crop_im)),
        reshape_4d(img2float(leye_im)),
        reshape_4d(img2float(reye_im)),
        reshape_4d(img2float(nose_im)),
        reshape_4d(img2float(mouth_im)),
        np.random.normal(size=(1, 100))
    ]
    generate_image, *_ = model.predict(inputs, batch_size=1)
    out_image = (generate_image[0] * 255).astype(np.uint8)
    return cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

# ===========================
# FLASK WEB APP
# ===========================
app = Flask(__name__)
MODEL_PATH = "D:/latest_dltpgan/generator/epoch7000_loss0.356.hdf5"
model = build_model(MODEL_PATH)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TP-GAN Web Demo</title>
<style>
body { font-family: Arial; text-align: center; margin-top: 40px; }
video, canvas, img { border: 1px solid #ccc; margin-top: 10px; }
button { margin-top: 10px; padding: 10px 20px; font-size: 16px; }
#container { position: relative; display: inline-block; }
#canvas { position: absolute; top: 0; left: 0; }
</style>
</head>
<body>
<h1>TP-GAN: Side Face → Frontal Face</h1>
<div id="container">
<video id="video" width="320" height="240" autoplay></video>
<canvas id="canvas" width="320" height="240"></canvas>
</div>
<br>
<button id="capture">Capture & Generate</button>
<h3>Generated Frontal Face:</h3>
<img id="result" src="" alt="Generated face" width="256">
<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const result = document.getElementById('result');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => { video.srcObject = stream; })
  .catch(err => console.error('Camera error:', err));

function drawGuide() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'lime';
  ctx.lineWidth = 2;
  const rectWidth = 128;
  const rectHeight = 128;
  ctx.strokeRect(
    canvas.width/2 - rectWidth/2, 
    canvas.height/2 - rectHeight/2, 
    rectWidth, 
    rectHeight
  );
  requestAnimationFrame(drawGuide);
}
drawGuide();

async function checkFaceAlignment() {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 128;
  tempCanvas.height = 128;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(video, video.videoWidth/2 - 64, video.videoHeight/2 - 64, 128, 128, 0, 0, 128, 128);
  const imgDataUrl = tempCanvas.toDataURL('image/jpeg');
  const res = await fetch('/check_alignment', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({image: imgDataUrl})
  });
  return await res.json();
}

document.getElementById('capture').addEventListener('click', async () => {
  const alignment = await checkFaceAlignment();
  if(!alignment.aligned){
    alert(alignment.message);
    return;
  }

  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = 128;
  captureCanvas.height = 128;
  const captureCtx = captureCanvas.getContext('2d');
  captureCtx.drawImage(video, video.videoWidth/2 - 64, video.videoHeight/2 - 64, 128, 128, 0, 0, 128, 128);
  const dataUrl = captureCanvas.toDataURL('image/jpeg');

  result.src = "";
  result.alt = "Generating...";
  const res = await fetch('/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl })
  });
  const json = await res.json();
  result.src = json.result;
  result.alt = "Generated frontal face";
});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/check_alignment', methods=['POST'])
def check_alignment():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        landmarks = fa.get_landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if landmarks is None:
            return jsonify({'aligned': False, 'message': 'No face detected. Please center your face in the rectangle.'})

        leye = np.average(landmarks[0][42:48], axis=0)
        reye = np.average(landmarks[0][36:42], axis=0)
        eye_center = (leye + reye) / 2
        angle = np.arctan2(reye[1]-leye[1], reye[0]-leye[0]) * 180 / np.pi

        if abs(angle) > 20:
            return jsonify({'aligned': False, 'message': 'Please keep your head more horizontal.'})
        if abs(eye_center[0]-64) > 20 or abs(eye_center[1]-64) > 20:
            return jsonify({'aligned': False, 'message': 'Please center your face inside the rectangle.'})

        return jsonify({'aligned': True, 'message': 'Face aligned!'})
    except Exception as e:
        return jsonify({'aligned': False, 'message': str(e)})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        side_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frontal = generate_frontal_face_full(model, side_img)
        _, buffer = cv2.imencode('.jpg', frontal)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'result': f"data:image/jpeg;base64,{jpg_as_text}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("🚀 TP-GAN Web Demo Running at: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)
