const config = window.APP_CONFIG || {
API_BASE_URL: "http://127.0.0.1:5000",
ENABLE_BACKEND: false
};

const webcam = document.getElementById("webcam");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const cameraMessage = document.getElementById("cameraMessage");

const startCameraBtn = document.getElementById("startCameraBtn");
const startInterviewBtn = document.getElementById("startInterviewBtn");
const stopInterviewBtn = document.getElementById("stopInterviewBtn");
const candidateForm = document.getElementById("candidateForm");

let cameraStream = null;
let interviewRunning = false;
let interviewStartTime = null;
let timerInterval = null;
let backendInterval = null;
let frameInterval = null;
let frameUploadInProgress = false;
let savedCandidate = null;
const analysisCanvas = document.createElement("canvas");
const analysisContext = analysisCanvas.getContext("2d");

function setText(id, value) {
const element = document.getElementById(id);
if (element) element.textContent = value;
}

function formatDuration(seconds) {
const hours = Math.floor(seconds / 3600);
const minutes = Math.floor((seconds % 3600) / 60);
const remainingSeconds = seconds % 60;

return [
    String(hours).padStart(2, "0"),
    String(minutes).padStart(2, "0"),
    String(remainingSeconds).padStart(2, "0")
].join(":");

}

function updateTimer() {
if (!interviewStartTime) return;

const elapsed = Math.floor(
    (Date.now() - interviewStartTime) / 1000
);

const duration = formatDuration(elapsed);
setText("timer", duration);
setText("finalDuration", duration);

}

function updateCameraUI(active) {
webcam.classList.toggle("visible", active);
cameraPlaceholder.classList.toggle("hidden", active);
startCameraBtn.disabled = active;

setText("cameraBadge", active ? "CAMERA ON" : "CAMERA OFF");

document.getElementById("cameraBadge")
    .classList.toggle("active", active);

}

async function startCamera() {
if (!navigator.mediaDevices ||
!navigator.mediaDevices.getUserMedia) {
cameraMessage.textContent =
"Camera unavailable. Open the project using localhost or HTTPS.";
return false;
}

try {
    cameraMessage.textContent =
        "Please allow camera permission in your browser.";

    cameraStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    });

    webcam.srcObject = cameraStream;
    await webcam.play();

    updateCameraUI(true);
    cameraMessage.textContent = "Camera is working.";
    return true;

} catch (error) {
    console.error("Camera error:", error);

    if (error.name === "NotAllowedError") {
        cameraMessage.textContent =
            "Camera permission denied. Allow camera access in browser settings.";
    } else if (error.name === "NotFoundError") {
        cameraMessage.textContent = "No camera was found.";
    } else {
        cameraMessage.textContent =
            "Could not start camera. Check permissions and close other camera apps.";
    }

    return false;
}

}

function stopCamera() {
if (cameraStream) {
cameraStream.getTracks().forEach(track => track.stop());
cameraStream = null;
}

webcam.srcObject = null;
updateCameraUI(false);
cameraMessage.textContent = "Camera is off.";

}

function resetAnalysis() {
setText("gazeValue", "—");
setText("behaviorValue", "—");
setText("expressionValue", "—");
setText("suspiciousValue", "—");

setText("gazeStatus", "Waiting for backend");
setText("behaviorStatus", "Waiting for backend");
setText("expressionStatus", "Waiting for backend");
setText("suspiciousStatus", "Waiting for backend");

["gazeStatus", "behaviorStatus", "expressionStatus", "suspiciousStatus"]
    .forEach(id => {
        document.getElementById(id).classList.remove("connected");
    });

}

function updateAnalysisCard(valueId, statusId, value, status) {
setText(valueId, value ?? "No result");
setText(statusId, status ?? "Result received");

document.getElementById(statusId).classList.add("connected");

}

async function checkBackend() {
if (!config.ENABLE_BACKEND) {
setText("connectionStatus", "Backend not connected");
setText("finalBackendStatus", "Disconnected");
return;
}

try {
    const response = await fetch(
        config.API_BASE_URL + "/api/health"
    );

    if (!response.ok) throw new Error("Health check failed");

    const data = await response.json();
    const connected = data.status === "ok";
    const ready = data.analysis_ready !== false;

    setText(
        "connectionStatus",
        connected && ready ? "Backend connected" :
            connected ? "Backend connected; model missing" : "Backend unavailable"
    );

    setText(
        "finalBackendStatus",
        connected && ready ? "Connected" :
            connected ? "Model missing" : "Unavailable"
    );

} catch (error) {
    console.error("Backend error:", error);
    setText("connectionStatus", "Backend unavailable");
    setText("finalBackendStatus", "Unavailable");
}

}

async function fetchAnalysisResults() {
if (!config.ENABLE_BACKEND || !interviewRunning) return;

try {
    const response = await fetch(
        config.API_BASE_URL + "/api/analysis/latest"
    );

    if (!response.ok) throw new Error("Analysis request failed");

    const data = await response.json();

    if (data.gaze) {
        updateAnalysisCard(
            "gazeValue", "gazeStatus",
            data.gaze.value, data.gaze.status
        );
    }

    if (data.behavior) {
        updateAnalysisCard(
            "behaviorValue", "behaviorStatus",
            data.behavior.value, data.behavior.status
        );
    }

    if (data.expression) {
        updateAnalysisCard(
            "expressionValue", "expressionStatus",
            data.expression.value, data.expression.status
        );
    }

    if (data.suspicious_activity) {
        updateAnalysisCard(
            "suspiciousValue", "suspiciousStatus",
            data.suspicious_activity.value,
            data.suspicious_activity.status
        );
    }

} catch (error) {
    console.error("Analysis results error:", error);
}

}

async function uploadAnalysisFrame() {
if (!config.ENABLE_BACKEND || !interviewRunning || !cameraStream || frameUploadInProgress) return;
if (!webcam.videoWidth || !webcam.videoHeight || !savedCandidate?.session_id) return;

frameUploadInProgress = true;
try {
    const scale = Math.min(1, 640 / webcam.videoWidth);
    analysisCanvas.width = Math.round(webcam.videoWidth * scale);
    analysisCanvas.height = Math.round(webcam.videoHeight * scale);
    analysisContext.drawImage(webcam, 0, 0, analysisCanvas.width, analysisCanvas.height);
    const frame = await new Promise(resolve => analysisCanvas.toBlob(resolve, "image/jpeg", 0.7));
    if (!frame) return;

    const formData = new FormData();
    formData.append("frame", frame, "frame.jpg");
    formData.append("session_id", savedCandidate.session_id);
    const response = await fetch(config.API_BASE_URL + "/api/analysis/frame", {
        method: "POST",
        body: formData
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Frame analysis failed");
    }
} catch (error) {
    console.error("Frame upload error:", error);
    cameraMessage.textContent = "Analysis paused: " + error.message;
} finally {
    frameUploadInProgress = false;
}
}

candidateForm.addEventListener("submit", function (event) {
event.preventDefault();

const name = document.getElementById("candidateName").value.trim();
const candidateId = document.getElementById("candidateId").value.trim();
const role = document.getElementById("interviewRole").value;

if (!name || !candidateId || !role) {
    alert("Please fill in all interview details.");
    return;
}

savedCandidate = {
    name: name,
    candidate_id: candidateId,
    role: role
};

setText("summaryName", name);
setText("summaryRole", role);

alert("Interview details saved successfully.");

});

startCameraBtn.addEventListener("click", startCamera);

startInterviewBtn.addEventListener("click", async function () {
if (interviewRunning) return;

if (!savedCandidate) {
    alert("Please save the candidate details first.");
    return;
}

if (!cameraStream) {
    const started = await startCamera();
    if (!started) return;
}

if (config.ENABLE_BACKEND) {
    try {
        const response = await fetch(config.API_BASE_URL + "/api/interview/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(savedCandidate)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Could not start backend session");
        savedCandidate.session_id = result.session_id;
    } catch (error) {
        console.error("Start session error:", error);
        cameraMessage.textContent = error.message;
        return;
    }
}

interviewRunning = true;
interviewStartTime = Date.now();

resetAnalysis();

setText("sessionStatus", "Interview in progress");
setText("summarySession", "In progress");
setText("finalSessionStatus", "In progress");

document.getElementById("cameraOverlay")
    .classList.add("visible");

startInterviewBtn.disabled = true;
stopInterviewBtn.disabled = false;

timerInterval = setInterval(updateTimer, 1000);

cameraMessage.textContent = "Interview started. Analysis is running.";

await checkBackend();

if (config.ENABLE_BACKEND) {
    uploadAnalysisFrame();
    frameInterval = setInterval(uploadAnalysisFrame, 1000);
    await fetchAnalysisResults();
    backendInterval = setInterval(fetchAnalysisResults, 2000);
}

});

stopInterviewBtn.addEventListener("click", async function () {
if (!interviewRunning) return;

interviewRunning = false;
updateTimer();

clearInterval(timerInterval);
clearInterval(backendInterval);
clearInterval(frameInterval);

timerInterval = null;
backendInterval = null;
frameInterval = null;

setText("sessionStatus", "Interview completed");
setText("summarySession", "Completed");
setText("finalSessionStatus", "Completed");

document.getElementById("cameraOverlay")
    .classList.remove("visible");

startInterviewBtn.disabled = false;
stopInterviewBtn.disabled = true;

cameraMessage.textContent =
    "Interview stopped. Camera remains on until you turn it off or close the page.";

if (config.ENABLE_BACKEND) {
    try {
        await fetch(
            config.API_BASE_URL + "/api/interview/stop",
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: savedCandidate.session_id })
            }
        );
    } catch (error) {
        console.error("Stop session error:", error);
    }
}

});

window.addEventListener("beforeunload", function () {
clearInterval(timerInterval);
clearInterval(backendInterval);
clearInterval(frameInterval);
if (interviewRunning && savedCandidate?.session_id) {
    const body = new Blob(
        [JSON.stringify({ session_id: savedCandidate.session_id })],
        { type: "application/json" }
    );
    navigator.sendBeacon(config.API_BASE_URL + "/api/interview/stop", body);
}
stopCamera();
});

checkBackend();