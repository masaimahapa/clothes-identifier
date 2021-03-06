import json
import urllib.request
import os
from flask import Flask, render_template
from flask import request
from fastai.vision import *

app= Flask(__name__)

UPLOADS_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER

def main():
    app.run()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST', 'GET'])
def predict():

    if request.method=="POST":
        print(request.files)
        if request.files:

            uploaded_image= request.files['image']
            image_name= uploaded_image.filename

            
            defaults.device= torch.device('cpu')
            path= Path('clothesurl/')
            learner= load_learner(path, 'export.pkl')
            
            img=open_image(uploaded_image)

            pred_class, pred_idx, outputs= learner.predict(img)

            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

            save_path=f'static/uploads/{image_name}'
            img.save(save_path)


    return render_template('prediction.html', pred_class=pred_class, save_path=full_filename)


if __name__ == "__main__":
    main()
    
