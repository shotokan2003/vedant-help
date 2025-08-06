# Body Tracking AI - CDAC DRONE TECH

## GitHub Repository
[https://github.com/shotokan2003/vedant-help.git](https://github.com/shotokan2003/vedant-help.git)

## Project Overview
This project provides a real-time body tracking AI system that can recognize different body postures and behaviors using a webcam. It uses a pre-trained machine learning model to detect and classify body positions in real-time.

## Features
- Real-time body posture detection and classification
- Live webcam visualization with MediaPipe landmark overlays
- Movement intensity tracking
- Behavior prediction
- Interactive web interface
- Graphical analysis of detection results
- Voice summary reports

## Requirements
To run this project, you need the following Python packages:
```
fastapi
uvicorn
opencv-python
mediapipe
numpy
pandas
joblib
matplotlib
gtts
playsound
```

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/shotokan2003/vedant.git
   cd vedant
   ```
2. Create and activate a new conda environment:
   ```
   conda create --prefix venv python=3.9.23
   conda activate .\venv
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Make sure the `Body_Tracking.pkl` model file is in the same directory as the code
2. Run the FastAPI application:
   ```
   python fixed_colab.py
   ```
3. Open your web browser and navigate to http://localhost:8000
4. Click "Start Tracking" to begin the body tracking process
5. Use the "Generate Graphs" button to see analysis results
6. Click "Play Voice Report" to hear a summary of the analysis

## Technical Details
- Model: Pre-trained machine learning model for body posture classification
- Frontend: HTML, CSS, JavaScript with WebSockets for real-time communication
- Backend: FastAPI for handling API requests and websocket connections
- Computer Vision: OpenCV and MediaPipe for body tracking and landmark detection

## License
This project is part of CDAC DRONE TECH and is intended for research and educational purposes.

## Credits
Developed as part of CDAC DRONE TECH initiative.
