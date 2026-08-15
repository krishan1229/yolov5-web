import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
import onnxruntime as ort

app = FastAPI(title="YOLOv5 Ultra-Light ONNX API")

# Load the ONNX model (Takes only ~40MB of RAM!)
cuda = False
providers = ['CPUExecutionProvider']
session = ort.InferenceSession("best.onnx", providers=providers)

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # Usually [1, 3, 640, 640]
img_size = input_shape[2] if len(input_shape) == 4 else 640

@app.get("/", response_class=HTMLResponse)
async def home_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv5 ONNX Cloud Detector</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 30px; background-color: #f4f4f9; text-align: center; }
            .container { background: white; padding: 30px; border-radius: 10px; display: inline-block; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); width: 85%; max-width: 1000px; }
            .controls { background: #eee; padding: 15px; border-radius: 8px; margin: 20px 0; display: flex; justify-content: center; gap: 40px; }
            .slider-group { display: flex; flex-direction: column; align-items: center; }
            .content-box { display: flex; flex-wrap: wrap; justify-content: center; gap: 30px; margin-top: 20px; }
            .view-panel { flex: 1; min-width: 300px; max-width: 500px; }
            button { background-color: #007BFF; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 5px; cursor: pointer; font-weight: bold; }
            button:hover { background-color: #0056b3; }
            #resultImg { width: 100%; display: none; border-radius: 5px; box-shadow: 0px 0px 5px rgba(0,0,0,0.2); }
            .text-panel { flex: 1; min-width: 300px; text-align: left; background: #fafafa; padding: 15px; border-radius: 5px; border: 1px solid #ddd; display: none; }
            .detection-item { background: white; padding: 10px; margin-bottom: 8px; border-radius: 4px; border-left: 5px solid #007BFF; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>YOLOv5 Object Detector (ONNX Low-Memory Cloud)</h2>
            <form id="uploadForm">
                <input type="file" id="imageInput" accept="image/*" required><br><br>
                <div class="controls">
                    <div class="slider-group">
                        <label><b>Conf Threshold:</b> <span id="confVal">0.25</span></label>
                        <input type="range" id="confSlider" min="0.05" max="1.0" step="0.05" value="0.25" oninput="document.getElementById('confVal').innerText=this.value">
                    </div>
                </div>
                <button type="submit">Analyze Image</button>
            </form>
            <div class="content-box">
                <div class="view-panel"><img id="resultImg" src="" alt="Detection Output"></div>
                <div class="text-panel" id="textPanel">
                    <h3>📊 Scaled Detection Details</h3>
                    <div id="detailsList"></div>
                </div>
            </div>
        </div>
        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('imageInput');
                const confSlider = document.getElementById('confSlider');
                const resultImg = document.getElementById('resultImg');
                const textPanel = document.getElementById('textPanel');
                const detailsList = document.getElementById('detailsList');
                if (fileInput.files.length === 0) return;
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                formData.append("conf_thresh", confSlider.value);
                try {
                    const response = await fetch('/predict-thresholds', { method: 'POST', body: formData });
                    if (response.ok) {
                        const data = await response.json();
                        resultImg.src = "data:image/jpeg;base64," + data.image;
                        resultImg.style.display = 'block';
                        detailsList.innerHTML = "";
                        if (data.predictions.length === 0) {
                            detailsList.innerHTML = "<p>No objects detected.</p>";
                        } else {
                            data.predictions.forEach((item, index) => {
                                const pct = (item.confidence * 100).toFixed(1);
                                detailsList.innerHTML += `
                                    <div class="detection-item">
                                        <strong>Object #${index + 1}:</strong> Object Class ID: ${item.class_id}<br>
                                        🎯 <strong>Confidence:</strong> ${pct}%<br>
                                        📍 <strong>Normalized Box (0 to 1):</strong><br>
                                        &nbsp;&nbsp;&bull; Xmin: ${item.xmin.toFixed(4)} &nbsp;|&nbsp; Ymin: ${item.ymin.toFixed(4)}<br>
                                        &nbsp;&nbsp;&bull; Xmax: ${item.xmax.toFixed(4)} &nbsp;|&nbsp; Ymax: ${item.ymax.toFixed(4)}
                                    </div>
                                `;
                            });
                        }
                        textPanel.style.display = 'block';
                    } else { alert('Error processing inference data.'); }
                } catch (err) { alert('Network error connecting to the API.'); }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict-thresholds")
async def predict_thresholds(file: UploadFile = File(...), conf_thresh: float = Form(0.25)):
    image_bytes = await file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = original_image.size

    # Resize and normalize image for ONNX input
    img = original_image.resize((img_size, img_size))
    img_data = np.array(img, dtype=np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))  # HWC to CHW
    img_data = np.expand_dims(img_data, axis=0)    # Add batch dimension

    # Run ONNX inference
    outputs = session.run(None, {input_name: img_data})
    output = outputs[0][0]  # Grab primary predictions matrix

    predictions = []
    draw = ImageDraw.Draw(original_image)

    # Parse bounding boxes
    for row in output:
        confidence = float(row[4])
        if confidence >= conf_thresh:
            # Get highest class score
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            
            if class_conf > 0.25:
                # Convert center x, center y, width, height format to 0-1 scale coordinates
                x_center, y_center, w, h = row[0]/img_size, row[1]/img_size, row[2]/img_size, row[3]/img_size
                xmin = max(0.0, x_center - w / 2)
                ymin = max(0.0, y_center - h / 2)
                xmax = min(1.0, x_center + w / 2)
                ymax = min(1.0, y_center + h / 2)

                predictions.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
                })

                # Draw bounding box using original raw pixel coordinates
                draw.rectangle([xmin * orig_w, ymin * orig_h, xmax * orig_w, ymax * orig_h], outline="red", width=4)

    # Encode image back to base64 string
    buffered = io.BytesIO()
    original_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": img_str, "predictions": predictions}
