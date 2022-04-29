import argparse
import io
import torch

from flask import Flask , request
from PIL import Image

app = Flask(__name__)

DETECTION_URL = '/v1/object-detection/yolov5s'

@app.route(DETECTION_URL , methods=['POST'])
def predict():
    if not request.method == 'POST':
        return 
    
    if request.file.get('image'):
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        
        results = model(im , size=640)
        return results.pandas().xyxy[0].to_json(orient="records")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    opt = parser.parse_args()

    
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat