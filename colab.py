import os
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import Counter
import uuid
import base64
import io
import json
import sys
from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Try to import mediapipe, but continue if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe successfully imported")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not available. Some features will be limited.")
    # Create placeholder for mp
    class PlaceholderMP:
        class solutions:
            class drawing_utils:
                DrawingSpec = lambda color, thickness, circle_radius: None
                
            class holistic:
                Holistic = lambda min_detection_confidence, min_tracking_confidence: None
                POSE_CONNECTIONS = []
                HAND_CONNECTIONS = []
                
                class PoseLandmark:
                    LEFT_SHOULDER = 0
                    RIGHT_SHOULDER = 1
                    LEFT_ELBOW = 2
                    RIGHT_ELBOW = 3
                    LEFT_KNEE = 4
                    RIGHT_KNEE = 5
                
            class face_mesh:
                FACEMESH_TESSELATION = []
    
    if not MEDIAPIPE_AVAILABLE:
        mp = PlaceholderMP()

# Initialize FastAPI app
app = FastAPI(title="Body Tracking AI", description="AI model for body tracking and behavior analysis")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Loading ML Model ===
try:
    model = joblib.load("Body_Tracking.pkl")
    MODEL_AVAILABLE = True
    print("ML model loaded successfully")
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"WARNING: Could not load model: {e}")
    model = None

# === Setting up features===
feature_names = [f'{coord}{i}' for i in range(1, 2005) for coord in ['x', 'y', 'z', 'v']][:2004]
label_map = {
    0: "standing_still", 1: "covering_face", 2: "right_hand_up", 3: "left_hand_up",
    4: "crossed_arms", 5: "fear_1", 6: "happy", 7: "melancholy", 8: "calling_out"
}

# === MediaPipe Setup ===
if MEDIAPIPE_AVAILABLE:
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh

    # === Styling ===
    neon_cyan = (0, 255, 255)
    neon_magenta = (255, 0, 255)
    neon_green = (0, 255, 0)

    face_landmark_style = mp_drawing.DrawingSpec(color=neon_cyan, thickness=1, circle_radius=2)
    face_connection_style = mp_drawing.DrawingSpec(color=neon_magenta, thickness=1, circle_radius=1)
    pose_style = mp_drawing.DrawingSpec(color=neon_green, thickness=3, circle_radius=4)
    pose_connection_style = mp_drawing.DrawingSpec(color=(200, 0, 255), thickness=2, circle_radius=2)
    hand_style = mp_drawing.DrawingSpec(color=neon_cyan, thickness=3, circle_radius=3)
    hand_connection_style = mp_drawing.DrawingSpec(color=neon_magenta, thickness=2, circle_radius=2)
else:
    # Set default values for styling when MediaPipe is not available
    neon_cyan = (0, 255, 255)
    neon_magenta = (255, 0, 255)
    neon_green = (0, 255, 0)

# === Data Storage ===
class SessionData:
    def __init__(self):
        self.prediction_history = []
        self.movement_scores_raw = []
        self.smoothed_scores = []
        self.previous_landmarks = None
        self.smoothing_window = 5
        self.frame_count = 0
        self.FRAME_RATE = 20.02  # Approximate frame rate

# Store session data per connection
sessions = {}

# === Serve static files ===
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# === HTML Response ===
@app.get("/", response_class=HTMLResponse)
async def get():
    with open('static/index.html', 'r') as f:
        return f.read()

# === WebSocket Connection ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionData()
    
    # Initialize holistic if mediapipe is available
    holistic = None
    if MEDIAPIPE_AVAILABLE:
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Process the frame
            img_data = data.split(",")[1]
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            session = sessions[session_id]
            session.frame_count += 1
            
            # Initialize variables
            behavior_prediction = "unknown"
            confidence = 0.0
            smoothed_value = 0.0
            
            # Process with MediaPipe if available
            if MEDIAPIPE_AVAILABLE and holistic:
                # Process the frame with MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                overlay = image.copy()

                # Draw landmarks
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(overlay, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                            face_landmark_style, face_connection_style)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(overlay, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                            hand_style, hand_connection_style)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(overlay, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                            hand_style, hand_connection_style)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(overlay, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                            pose_style, pose_connection_style)

                image = cv2.addWeighted(image, 1.0, overlay, 0.65, 0)
                
                # Movement Intensity
                movement_score = 0.0
                key_indices = [
                    mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_SHOULDER,
                    mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.RIGHT_ELBOW,
                    mp_holistic.PoseLandmark.LEFT_KNEE, mp_holistic.PoseLandmark.RIGHT_KNEE
                ]
                current_landmarks = [
                    results.pose_landmarks.landmark[i]
                    for i in key_indices
                    if results.pose_landmarks and results.pose_landmarks.landmark[i].visibility > 0.6
                ]

                if session.previous_landmarks and len(current_landmarks) == len(session.previous_landmarks):
                    distances = [
                        np.linalg.norm(np.array([c.x, c.y]) - np.array([p.x, p.y]))
                        for c, p in zip(current_landmarks, session.previous_landmarks)
                    ]
                    distances = [d for d in distances if d > 0.01]
                    movement_score = round(sum(distances) * 1000, 2)

                session.previous_landmarks = current_landmarks
                session.movement_scores_raw.append(movement_score)
                window = session.movement_scores_raw[-session.smoothing_window:]
                smoothed_value = round(np.mean(window), 2) if window else 0.0
                session.smoothed_scores.append(smoothed_value)
                
                # Behavior Prediction if model is available
                if MODEL_AVAILABLE and model and results.pose_landmarks:
                    try:
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten())
                        face = results.face_landmarks.landmark if results.face_landmarks else []
                        face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face]).flatten())
                        row = pose_row + face_row
                        row = row[:len(feature_names)]
                        while len(row) < len(feature_names):
                            row.append(0.0)

                        X = pd.DataFrame([row], columns=feature_names)
                        pred_class = model.predict(X)[0]
                        behavior_prediction = label_map.get(pred_class, str(pred_class))
                        
                        # Add prediction probabilities if the model supports it
                        try:
                            proba = model.predict_proba(X)[0]
                            confidence = float(proba[pred_class])
                        except:
                            confidence = 1.0  # Default confidence
                        
                        session.prediction_history.append(behavior_prediction)
                        
                        # Add behavior prediction text to the frame
                        label_bg = (245, 117, 16)
                        cv2.rectangle(image, (10, 10), (10 + len(behavior_prediction)*20, 50), label_bg, -1)
                        cv2.putText(image, behavior_prediction, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        session.prediction_history.append("error")
            else:
                # If MediaPipe is not available, just use the original frame with minimal processing
                image = frame.copy()
                cv2.putText(image, "MediaPipe not available", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
                # Add random movement values for demo purposes
                movement_score = np.random.random() * 10
                session.movement_scores_raw.append(movement_score)
                window = session.movement_scores_raw[-session.smoothing_window:]
                smoothed_value = round(np.mean(window), 2) if window else 0.0
                session.smoothed_scores.append(smoothed_value)
                session.prediction_history.append("demo_mode")
                behavior_prediction = "demo_mode"
            
            # Add border around the frame
            cv2.rectangle(image, (5, 5), (635, 475), (0, 190, 255), thickness=2)
            
            # Convert the processed image back to base64 to send to the client
            _, buffer = cv2.imencode('.jpg', image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare analysis data to send to client
            analysis_data = {
                "processedFrame": f"data:image/jpeg;base64,{img_str}",
                "behavior": behavior_prediction,
                "confidence": confidence,
                "movementScore": smoothed_value,
                "frameCount": session.frame_count,
                "mediapipeAvailable": MEDIAPIPE_AVAILABLE,
                "modelAvailable": MODEL_AVAILABLE
            }
            
            await websocket.send_json(analysis_data)
            
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if session_id in sessions:
            del sessions[session_id]
        if MEDIAPIPE_AVAILABLE and holistic:
            holistic.close()

# === API Endpoints for Data Analysis ===
@app.post("/generate-graphs")
async def generate_graphs(session_id: dict):
    # Extract session_id from request body
    try:
        if isinstance(session_id, dict):
            session_id = session_id.get("session_id", "")
        elif isinstance(session_id, str):
            # If it's already a string, keep it as is
            pass
        else:
            # Convert any other type to string
            session_id = str(session_id)
        
        print(f"Received session ID: {session_id}")
        
        if not session_id or session_id not in sessions:
            # If no valid session ID, use a demo session
            print("Session not found, using demo data")
            # Create a demo session with random data
            demo_session = SessionData()
            demo_session.prediction_history = ["demo_mode", "standing_still", "right_hand_up", "demo_mode", 
                                            "left_hand_up", "demo_mode", "standing_still", "demo_mode"] * 5
            demo_session.smoothed_scores = [np.random.random() * 10 for _ in range(40)]
            session = demo_session
        else:
            session = sessions[session_id]
    except Exception as e:
        print(f"Error processing session ID: {e}")
        # Create demo data on error
        demo_session = SessionData()
        demo_session.prediction_history = ["error_mode", "standing_still", "error_mode"] * 10
        demo_session.smoothed_scores = [np.random.random() * 5 for _ in range(30)]
        session = demo_session
    
    # Only generate graphs if we have data
    if not session.prediction_history:
        return {"error": "No prediction data available"}
    
    graph_data = {}
    
    # Action Frequency Graph
    counts = Counter(session.prediction_history)
    labels = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(9, 4.5))
    plt.style.use('dark_background')
    bars = plt.bar(labels, values, color='#00FFE3', edgecolor='cyan', linewidth=2)
    plt.title("Action Frequency", fontsize=16, fontweight='bold', color='#00FFE3')
    plt.xlabel("Detected Action", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, str(height),
                 ha='center', va='bottom', fontsize=10, color='white')
    plt.tight_layout()
    
    # Save the figure to a base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    action_freq_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    graph_data["actionFrequency"] = action_freq_img
    
    # Behavior Timeline Graph
    plt.figure(figsize=(12, 3.5))
    plt.style.use('dark_background')
    plt.plot(session.prediction_history, marker='X', markersize=7, linestyle='--', linewidth=2.5,
             color='#FF00FF', markerfacecolor='black', markeredgecolor='#FF00FF')
    plt.title("Behavior Prediction Timeline", fontsize=16, fontweight='bold', color='#FF00FF')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.25)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    timeline_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    graph_data["behaviorTimeline"] = timeline_img
    
    # Movement Intensity Graph
    if session.smoothed_scores:
        x_vals = range(len(session.smoothed_scores))
        plt.figure(figsize=(10, 3.5))
        plt.style.use('dark_background')
        plt.plot(x_vals, session.smoothed_scores, color='#32CD32', linewidth=3)
        plt.fill_between(x_vals, session.smoothed_scores, alpha=0.3, color='#32CD32')
        plt.title("Smoothed Movement Intensity", fontsize=16, fontweight='bold', color='#32CD32')
        plt.xlabel("Frame Number", fontsize=12)
        plt.ylabel("Intensity", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.25)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        movement_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        graph_data["movementIntensity"] = movement_img
    
    # Generate summary text
    counts = Counter(session.prediction_history)
    most_frequent = max(counts, key=counts.get)
    avg_intensity = np.mean(session.smoothed_scores)
    peak_intensity = max(session.smoothed_scores)
    peak_index = session.smoothed_scores.index(peak_intensity)
    total_frames = len(session.smoothed_scores)
    total_time = total_frames / session.FRAME_RATE
    peak_time = peak_index / session.FRAME_RATE
    
    summary = {
        "totalFrames": total_frames,
        "totalTime": round(total_time, 2),
        "mostFrequentBehavior": most_frequent,
        "averageIntensity": round(avg_intensity, 2),
        "peakIntensity": round(peak_intensity, 2),
        "peakTime": round(peak_time, 2)
    }
    
    graph_data["summary"] = summary
    
    return graph_data

# === Run the FastAPI app ===
if __name__ == "__main__":
    uvicorn.run("colab:app", host="localhost", port=8000, reload=True)
