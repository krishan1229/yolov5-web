import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
import onnxruntime as ort
import gc

app = FastAPI(title="YOLOv5 Zero-Framework ONNX API")

# Initialize the ONNX session with disabled memory optimization structures to save space
opts = ort.SessionOptions()
opts.enable_mem_pattern = False
opts.enable_cpu_mem_arena = False
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

session = ort.InferenceSession("best.onnx", sess_options=opts, providers=['CPUExecutionProvider'])
input_name = session.get_inputs().name

# --- Pure NumPy Non-Maximum Suppression (NMS) Function ---
def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

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
                    <!-- Added IOU Threshold Slider back to interface -->
                    <div class="slider-group">
                        <label><b>IOU Threshold (Overlap Control):</b> <span id="iouVal">0.45</span></label>
                        <input type="range" id="iouSlider" min="0.05" max="1.0" step="0.05" value="0.45" oninput="document.getElementById('iouVal').innerText=this.value">
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
                const iouSlider = document.getElementById('iouSlider');
                const resultImg = document.getElementById('resultImg');
                const textPanel = document.getElementById('textPanel');
                const detailsList = document.getElementById('detailsList');
                if (fileInput.files.length === 0) return;
                
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                formData.append("conf_thresh", confSlider.value);
                formData.append("iou_thresh", iouSlider.value);
                
                try {
                    const response = await fetch('/predict-thresholds', { method: 'POST', body: formData });
                    if (response.ok) {
                        const data = await response.json();
                        resultImg.src = "data:image/jpeg;base64," + data.image;
                        resultImg.style.display = 'block';
                        detailsList.innerHTML = "";
                        if (data.predictions.length === 0) {
                            detailsList.innerHTML = "<p>No objects detected with current thresholds.</p>";
                        } else {
                            data.predictions.forEach((item, index) => {
                                const pct = (item.confidence * 100).toFixed(1);
                                detailsList.innerHTML += `
                                    <div class="detection-item">
                                        <strong>Object #${index + 1}:</strong> Class ID ${item.class_id}<br>
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
async def predict_thresholds(
    file: UploadFile = File(...), 
    conf_thresh: float = Form(0.25),
    iou_thresh: float = Form(0.45)
):
    image_bytes = await file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = original_image.size

    img = original_image.resize((224, 224))
    img_data = np.array(img, dtype=np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    outputs = session.run(None, {input_name: img_data})
    output = np.squeeze(outputs)

    candidate_boxes = []
    candidate_scores = []
    candidate_classes = []

    if len(output.shape) == 2:
        for row in output:
            objectness = row[4]
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            confidence = float(objectness * class_scores[class_id])

            if confidence >= conf_thresh:
                x_center, y_center, w, h = row[0], row[1], row[2], row[3]
                
                x1 = max(0.0, x_center - w / 2)
                y1 = max(0.0, y_center - h / 2)
                x2 = min(224.0, x_center + w / 2)
                y2 = min(224.0, y_center + h / 2)

                candidate_boxes.append([x1, y1, x2, y2])
                candidate_scores.append(confidence)
                candidate_classes.append(class_id)

    predictions = []
    draw = ImageDraw.Draw(original_image)

    # Apply NMS calculations to clean up overlapping candidate boxes
    if len(candidate_boxes) > 0:
        np_boxes = np.array(candidate_boxes, dtype=np.float32)
        np_scores = np.array(candidate_scores, dtype=np.float32)
        
        keep_indices = nms(np_boxes, np_scores, iou_thresh)
        
        for idx in keep_indices:
            box224 = candidate_boxes[idx]
            confidence = candidate_scores[idx]
            class_id = candidate_classes[idx]

            xmin, ymin = float(box224[0] / 224.0), float(box224[1] / 224.0)
            xmax, ymax = float(box224[2] / 224.0), float(box224[3] / 224.0)

            predictions.append({
                "class_id": int(class_id),
                "confidence": float(confidence),
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
            })

            # UPDATED STYLE: Bounding box drawn in blue with width 1
            draw.rectangle(
