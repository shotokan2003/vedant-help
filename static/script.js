// Global Variables
let webSocket = null;
let sessionId = null;
let isStreaming = false;
let videoStream = null;
let speechSynthesis = window.speechSynthesis;

// DOM Elements
const webcamVideo = document.getElementById("webcam");
const overlayCanvas = document.getElementById("overlay");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const generateGraphsBtn = document.getElementById("generate-graphs");
const voiceReportBtn = document.getElementById("voice-report");
const connectionStatus = document.getElementById("connection-status");
const statusDot = document.getElementById("status-dot");
const behaviorValue = document.getElementById("behavior-value");
const confidenceValue = document.getElementById("confidence-value");
const movementValue = document.getElementById("movement-value");
const frameCountElement = document.getElementById("frame-count");
const graphsContainer = document.getElementById("graphs-container");

// Performance optimization variables
let processingFrame = false;
let lastFrameTime = 0;
const TARGET_FPS = 15; // Reduced from 10fps for better responsiveness
const FRAME_INTERVAL = 1000 / TARGET_FPS;
let frameQueue = [];
const MAX_QUEUE_SIZE = 2; // Limit queue to prevent backlog

// Event Listeners
document.addEventListener("DOMContentLoaded", initializeApp);
startButton.addEventListener("click", startTracking);
stopButton.addEventListener("click", stopTracking);
generateGraphsBtn.addEventListener("click", generateGraphs);
voiceReportBtn.addEventListener("click", playVoiceReport);

// Initialize the application
function initializeApp() {
  checkCameraPermission();
  updateConnectionStatus(false);
}

// Check if we have camera permissions
async function checkCameraPermission() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    stream.getTracks().forEach((track) => track.stop());
    console.log("Camera permission granted");
  } catch (err) {
    console.error("Camera permission denied:", err);
    alert("Camera permission is required for this application to work.");
  }
}

// Update the connection status indicator
function updateConnectionStatus(isConnected) {
  if (isConnected) {
    connectionStatus.textContent = "Connected";
    statusDot.classList.add("connected");
  } else {
    connectionStatus.textContent = "Disconnected";
    statusDot.classList.remove("connected");
  }
}

// Generate UUID for session
function generateUUID() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    var r = (Math.random() * 16) | 0,
      v = c == "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Start tracking and connect to the WebSocket server
async function startTracking() {
  try {
    // Get camera stream
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640, max: 640 },
        height: { ideal: 480, max: 480 },
        facingMode: "user",
        frameRate: { ideal: 30, max: 30 }, // Higher camera fps for smoother display
      },
    });
    webcamVideo.srcObject = videoStream;

    // Generate session ID
    sessionId = generateUUID();
    console.log("Generated session ID:", sessionId);

    // Connect to WebSocket
    const wsUrl = `ws://localhost:8000/ws`;
    console.log("Attempting WebSocket connection to:", wsUrl);
    webSocket = new WebSocket(wsUrl);

    webSocket.onmessage = (event) => {
      // Handle ping messages to keep connection alive
      if (event.data === "ping") {
        console.log("Received ping, sending pong");
        webSocket.send("pong");
        return;
      }

      // Handle data messages
      try {
        const data = JSON.parse(event.data);
        // Use requestAnimationFrame for smooth updates
        requestAnimationFrame(() => {
          updateOverlay(data);
          updateMetrics(data);
        });
      } catch (err) {
        console.log("Received non-JSON message:", event.data);
      }
    };

    webSocket.onclose = (event) => {
      console.log("WebSocket connection closed:", event.code, event.reason);
      updateConnectionStatus(false);
      stopTracking();

      // Automatically reconnect if connection was lost unexpectedly
      if (event.code === 1006 || event.code === 1011) {
        console.log("Attempting to reconnect in 3 seconds...");
        setTimeout(startTracking, 3000);
      }
    };

    webSocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      alert("Connection error. Please check the server and try again.");
      stopTracking();
    };

    // Send session ID to server after WebSocket connection established
    webSocket.onopen = () => {
      console.log("✓ WebSocket connection established");
      updateConnectionStatus(true);
      isStreaming = true;

      // Send session ID to server immediately
      webSocket.send(
        JSON.stringify({ type: "session_init", session_id: sessionId })
      );
      console.log("Sent session ID to server:", sessionId);

      startVideoProcessing();
      startButton.disabled = true;
      stopButton.disabled = false;
      generateGraphsBtn.disabled = false;
    };
  } catch (err) {
    console.error("Error starting tracking:", err);
    alert("Failed to access camera: " + err.message);
  }
}

// Stop tracking and close connections
function stopTracking() {
  isStreaming = false;
  processingFrame = false;
  frameQueue = [];

  // Close WebSocket
  if (webSocket && webSocket.readyState === WebSocket.OPEN) {
    webSocket.close();
  }

  // Stop video tracks
  if (videoStream) {
    videoStream.getTracks().forEach((track) => track.stop());
    videoStream = null;
  }

  // Reset UI
  webcamVideo.srcObject = null;
  updateConnectionStatus(false);
  startButton.disabled = false;
  stopButton.disabled = true;

  // Clear overlay
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// Start processing video frames and sending to server
function startVideoProcessing() {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext("2d");

  // Set the correct size for the overlay canvas
  overlayCanvas.width = 640;
  overlayCanvas.height = 480;

  // Make sure video is playing
  webcamVideo.play().catch((err) => {
    console.error("Error playing video:", err);
    alert("Could not play webcam video. Please check your camera permissions.");
    stopTracking();
    return;
  });

  // Wait for video metadata to load
  if (webcamVideo.readyState < 2) {
    // HAVE_CURRENT_DATA or higher
    console.log("Waiting for video to be ready...");
    webcamVideo.addEventListener("loadeddata", () => {
      console.log("Video ready, starting processing");
      startFrameProcessing();
    });
  } else {
    startFrameProcessing();
  }

  function startFrameProcessing() {
    function processFrame(timestamp) {
      if (!isStreaming) return;

      // Improved frame rate control
      if (timestamp - lastFrameTime >= FRAME_INTERVAL && !processingFrame) {
        processingFrame = true;
        lastFrameTime = timestamp;

        try {
          if (webcamVideo.readyState >= 2) {
            // Draw video frame to hidden canvas
            ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);

            // Only send frame if WebSocket is ready and queue isn't full
            if (
              webSocket &&
              webSocket.readyState === WebSocket.OPEN &&
              frameQueue.length < MAX_QUEUE_SIZE
            ) {
              try {
                // Higher quality for better tracking accuracy vs performance balance
                const imageData = canvas.toDataURL("image/jpeg", 0.8);

                // Add to queue instead of sending immediately
                frameQueue.push(imageData);

                // Process queue asynchronously
                if (frameQueue.length === 1) {
                  processFrameQueue();
                }
              } catch (err) {
                console.error("Error creating frame data:", err);
              }
            }
          }
        } catch (err) {
          console.error("Error processing frame:", err);
        } finally {
          processingFrame = false;
        }
      }

      // Continue the processing loop
      requestAnimationFrame(processFrame);
    }

    // Start the processing loop
    requestAnimationFrame(processFrame);
  }

  // Asynchronous frame queue processing
  async function processFrameQueue() {
    while (
      frameQueue.length > 0 &&
      webSocket &&
      webSocket.readyState === WebSocket.OPEN
    ) {
      const frameData = frameQueue.shift();
      try {
        webSocket.send(frameData);
      } catch (err) {
        console.error("Error sending frame:", err);
        break;
      }

      // Small delay to prevent overwhelming the connection
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
  }
}

// Optimized overlay update
function updateOverlay(data) {
  if (!data.processedFrame) return;

  // Use faster image loading approach
  const ctx = overlayCanvas.getContext("2d");
  const img = new Image();

  img.onload = () => {
    // Clear and draw in one operation
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    ctx.drawImage(img, 0, 0, overlayCanvas.width, overlayCanvas.height);
  };

  img.onerror = () => {
    console.error("Failed to load processed frame");
  };

  img.src = data.processedFrame;
}

// Optimized metrics update with throttling
let lastMetricsUpdate = 0;
const METRICS_UPDATE_INTERVAL = 100; // Update metrics every 100ms

function updateMetrics(data) {
  const now = Date.now();
  if (now - lastMetricsUpdate < METRICS_UPDATE_INTERVAL) {
    return; // Skip update if too frequent
  }
  lastMetricsUpdate = now;

  behaviorValue.textContent = data.behavior || "N/A";
  confidenceValue.textContent = data.confidence
    ? `${(data.confidence * 100).toFixed(1)}%`
    : "0%";
  movementValue.textContent = data.movementScore
    ? data.movementScore.toFixed(2)
    : "0.00";
  frameCountElement.textContent = data.frameCount || "0";

  // Add some color coding based on behavior
  behaviorValue.style.color = getBehaviorColor(data.behavior);
}

// Get color based on behavior
function getBehaviorColor(behavior) {
  const behaviorColors = {
    standing_still: "#FFFFFF",
    covering_face: "#FFCC00",
    right_hand_up: "#66CCFF",
    left_hand_up: "#66FFCC",
    crossed_arms: "#FF6666",
    fear_1: "#FF3333",
    happy: "#33FF33",
    melancholy: "#9966FF",
    calling_out: "#FF66FF",
  };

  return behaviorColors[behavior] || "#FFFFFF";
}

// Generate graphs based on collected data
async function generateGraphs() {
  if (!sessionId) {
    alert("No active session. Please start tracking first.");
    return;
  }

  try {
    // Show loading indicator
    graphsContainer.innerHTML =
      '<div class="loading">Generating graphs...</div>';
    graphsContainer.style.display = "block";

    console.log("Requesting graphs for session ID:", sessionId);

    // Request graphs from server
    const response = await fetch("/generate-graphs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ session_id: sessionId }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      alert(`Error: ${data.error}`);
      return;
    }
    // Update HTML content with graph data
    const graphHTML = `
            <h2>Analysis Results</h2>
            <div class="graph-wrapper">
                <h3>Action Frequency</h3>
                <div class="graph-image-container">
                    <img id="action-frequency-graph" src="data:image/png;base64,${data.actionFrequency}" alt="Action Frequency Graph">
                </div>
            </div>
            <div class="graph-wrapper">
                <h3>Behavior Timeline</h3>
                <div class="graph-image-container">
                    <img id="behavior-timeline-graph" src="data:image/png;base64,${data.behaviorTimeline}" alt="Behavior Timeline Graph">
                </div>
            </div>
            <div class="graph-wrapper">
                <h3>Movement Intensity</h3>
                <div class="graph-image-container">
                    <img id="movement-intensity-graph" src="data:image/png;base64,${data.movementIntensity}" alt="Movement Intensity Graph">
                </div>
            </div>
            
            <div class="summary-panel">
                <h3>Analysis Summary</h3>
                <div id="summary-text"></div>
                <button id="voice-report">Play Voice Report</button>
            </div>
        `;

    // Show and update the graphs container
    graphsContainer.innerHTML = graphHTML;
    graphsContainer.style.display = "block";

    // Update summary text
    const summary = data.summary;
    const summaryText = `
            <p>Total frames analyzed: <strong>${summary.totalFrames}</strong></p>
            <p>Total tracking duration: <strong>${summary.totalTime} seconds</strong></p>
            <p>Most frequent behavior: <strong>${summary.mostFrequentBehavior}</strong></p>
            <p>Average movement intensity: <strong>${summary.averageIntensity}</strong></p>
            <p>Peak intensity: <strong>${summary.peakIntensity}</strong> at <strong>${summary.peakTime} seconds</strong></p>
        `;

    const summaryElement = document.getElementById("summary-text");
    if (summaryElement) {
      summaryElement.innerHTML = summaryText;
    }

    // Re-attach the event listener for the voice report button
    const voiceReportBtn = document.getElementById("voice-report");
    if (voiceReportBtn) {
      voiceReportBtn.addEventListener("click", playVoiceReport);
    }
  } catch (err) {
    console.error("Error generating graphs:", err);
    alert("Failed to generate graphs: " + err.message);
  }
}

// Play the voice report based on summary data
function playVoiceReport() {
  const summaryElement = document.getElementById("summary-text");
  if (!summaryElement.textContent.trim()) {
    alert("No summary data available. Please generate graphs first.");
    return;
  }

  // Extract data from summary
  const summaryText = summaryElement.textContent;

  // Create voice message
  const voiceMessage = `
        Hello commander. Your Voice Assistant Here.
        ${summaryText.replace(/\s+/g, " ").trim()}
        Graphical diagnostics completed successfully.
        Visual and movement logs stored.
        Engaging systems on standby.
    `;

  // Use Speech Synthesis API
  const utterance = new SpeechSynthesisUtterance(voiceMessage);
  utterance.rate = 0.9; // Slightly slower
  utterance.pitch = 1.0;

  // Stop any current speech
  speechSynthesis.cancel();

  // Speak
  speechSynthesis.speak(utterance);
}
