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
input_name = session.get_inputs()[0].name

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
                            detailsList.innerHTML = "<p>No objects detected with current threshold.</p>";
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
async def predict_thresholds(file: UploadFile = File(...), conf_thresh: float = Form(0.25)):
    image_bytes = await file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = original_image.size

    # Resize input matching your local 224 structural transformation format
    img = original_image.resize((224, 224))
    img_data = np.array(img, dtype=np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    # Execute math session
    outputs = session.run(None, {input_name: img_data})
    output = np.squeeze(outputs[0])  # Access raw prediction matrix layer

    predictions = []
    draw = ImageDraw.Draw(original_image)

    # Standard loop to match YOLOv5 outputs shapes directly
    if len(output.shape) == 2:
        for row in output:
            objectness = row[4]
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            confidence = float(objectness * class_scores[class_id])

            if confidence >= conf_thresh:
                x_center, y_center, w, h = row[0], row[1], row[2], row[3]
                
                # Convert boundaries safely relative to 224 grid limits
                x1 = max(0.0, x_center - w / 2)
                y1 = max(0.0, y_center - h / 2)
                x2 = min(224.0, x_center + w / 2)
                y2 = min(224.0, y_center + h / 2)

                # Generate clean ratios (0 to 1) and force them to standard python floats
                xmin, ymin = float(x1 / 224.0), float(y1 / 224.0)
                xmax, ymax = float(x2 / 224.0), float(y2 / 224.0)
                
                predictions.append({
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "xmin": xmin, 
                    "ymin": ymin, 
                    "xmax": xmax, 
                    "ymax": ymax
                })


                draw.rectangle([xmin * orig_w, ymin * orig_h, xmax * orig_w, ymax * orig_h], outline="blue", width=1)

    buffered = io.BytesIO()
    original_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    del original_image, image_bytes, img_data, outputs, output
    gc.collect()

    return {"image": img_str, "predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
