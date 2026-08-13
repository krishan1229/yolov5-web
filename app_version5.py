from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import torch
from PIL import Image, ImageDraw
import io
import base64

app = FastAPI(title="YOLOv5 Normalized Output API")

# Load model offline from your local windows cache
model = torch.hub.load(r'C:\Users\krish\.cache\torch\hub\ultralytics_yolov5_master', 'custom', path='best.pt', source='local')

@app.get("/", response_class=HTMLResponse)
async def home_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv5 Scaled Detector</title>
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
            <h2>YOLOv5 Object Detector (Scaled Coordinates 0 to 1)</h2>
            
            <form id="uploadForm">
                <input type="file" id="imageInput" accept="image/*" required><br><br>
                
                <div class="controls">
                    <div class="slider-group">
                        <label><b>Conf Threshold:</b> <span id="confVal">0.25</span></label>
                        <input type="range" id="confSlider" min="0.05" max="1.0" step="0.05" value="0.25" oninput="document.getElementById('confVal').innerText=this.value">
                    </div>
                    <div class="slider-group">
                        <label><b>IOU Threshold:</b> <span id="iouVal">0.45</span></label>
                        <input type="range" id="iouSlider" min="0.05" max="1.0" step="0.05" value="0.45" oninput="document.getElementById('iouVal').innerText=this.value">
                    </div>
                </div>
                
                <button type="submit">Analyze Image</button>
            </form>
            
            <div class="content-box">
                <div class="view-panel">
                    <img id="resultImg" src="" alt="Detection Output">
                </div>
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
                // FIX: Grabbed index [0] to resolve the 422 submission block safely
                formData.append("file", fileInput.files[0]);
                formData.append("conf_thresh", confSlider.value);
                formData.append("iou_thresh", iouSlider.value);

                try {
                    const response = await fetch('/predict-thresholds', {
                        method: 'POST',
                        body: formData
                    });

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
                                        <strong>Object #${index + 1}:</strong> ${item.name}<br>
                                        🎯 <strong>Confidence:</strong> ${pct}%<br>
                                        📍 <strong>Normalized Box (0 to 1):</strong><br>
                                        &nbsp;&nbsp;&bull; Xmin: ${item.xmin.toFixed(4)} &nbsp;|&nbsp; Ymin: ${item.ymin.toFixed(4)}<br>
                                        &nbsp;&nbsp;&bull; Xmax: ${item.xmax.toFixed(4)} &nbsp;|&nbsp; Ymax: ${item.ymax.toFixed(4)}
                                    </div>
                                `;
                            });
                        }
                        textPanel.style.display = 'block';
                    } else {
                        alert('Error processing inference data on the server.');
                    }
                } catch (err) {
                    alert('Network error connecting to the API.');
                }
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
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Set parameters
    model.conf = conf_thresh
    model.iou = iou_thresh

    # Run inference
    results = model(image)
    
    # FIX: Isolated index [0] on both datasets to prevent index crashes
    pred_pixels = results.pandas().xyxy[0].to_dict(orient="records")
    pred_scaled = results.pandas().xyxyn[0].to_dict(orient="records")

    # Manually draw boxes using pixel data structures
    draw = ImageDraw.Draw(image)
    for pred in pred_pixels:
        box = [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']]
        draw.rectangle(box, outline="blue", width=1)

    # Encode modified photo
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "image": img_str,
        "predictions": pred_scaled
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
