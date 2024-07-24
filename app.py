from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import mediapipe as mp
import numpy as np
import pandas as pd
import winsound

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Inilisiasi mediapipe untuk deteksi
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Menyimpan informasi deteksi
detection_info = {
    "class": "-",
    "probability": "-"
}
# fungsi menghasilkan frame video
def generate_frames():
    global detection_info
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                row = pose_row + face_row

                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                max_prob = round(body_language_prob[np.argmax(body_language_prob)], 2)

                # Update global detection info
                detection_info["class"] = body_language_class
                detection_info["probability"] = str(max_prob)

                coords = tuple(np.multiply(
                    np.array(
                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                    , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(body_language_class)*20, coords[1]-30), 
                            (30, 81, 40), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (216, 233, 168), 2, cv2.LINE_AA)
                
                cv2.rectangle(image, (0,0), (250, 60), (30, 81, 40), -1)
                
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (216, 233, 168), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (216, 233, 168), 2, cv2.LINE_AA)
                
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (216, 233, 168), 1, cv2.LINE_AA)
                cv2.putText(image, str(max_prob)
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (216, 233, 168), 2, cv2.LINE_AA)
                
                # Jika terdeteksi "Poor Posture", bunyikan beep
                if body_language_class == "Poor Posture":
                    winsound.Beep(1000, 500)  # Frekuensi 1000 Hz selama 500 ms

            except:
                pass
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

# Mengirim hasil stream video dari fungsi generate_frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_info')
def detection_info_endpoint():
    return jsonify(detection_info)

if __name__ == '__main__':
    app.run(debug=True)
