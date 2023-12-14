from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from PIL import Image
from PIL import ImageOps
from matplotlib import cm
import numpy as np
from flask import url_for


app = Flask(__name__)

# Load the Keras model

modelR = load_model('my_model_3DEC.h5')

# Set up upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get_output():
    if 'img_file' not in request.files:
        return render_template("index.html", prediction="No file selected")

    img = request.files['img_file']

    if img.filename == '':
        return render_template("index.html", prediction="No file selected")

    if img and allowed_file(img.filename):

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img.filename))
        print(img_path)
        img.save(img_path)
        
        

        # Preprocesing
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))
        inverted_img = ImageOps.invert(image)
        img_array = np.array(inverted_img, dtype=np.float32) / 255.0
        img_array = img_array.reshape((1, 28, 28, 1))

        # pred 
        print(img_array)
        prediction = modelR.predict(img_array)
        predicted_class = np.argmax(prediction)
        print(prediction)
        prediction = prediction.flatten()
        result = f"The predicted number is: {predicted_class}"
        total = np.sum(prediction)
        
        confidence_percentage = 100 - round((prediction[predicted_class] / total) * 100, 2)
        confidence = f"I am {confidence_percentage} % Confident!"

# Add a new route to serve uploaded images



        uploaded_image = url_for('uploaded_file', filename=secure_filename(img.filename))
        print(os.path.basename(uploaded_image))
        
        # Print the confidence percentage
        # print(f'Confidence Percentage: {confidence_percentage:.2f}%')
        return render_template("index.html", prediction=result, confidence = confidence, uploaded_image=os.path.basename(uploaded_image))
    else:
        return render_template("index.html", prediction="Invalid file type")

if __name__ == '__main__':
    app.run(debug=True)














