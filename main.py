import os
from flask import Flask, render_template, abort, redirect, url_for, request
from werkzeug.utils import secure_filename
app = Flask(__name__)

UPLOAD_FOLDER = 'upload'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in set(['jpg', 'png', 'jpeg'])

@app.route("/upload", methods=['POST'])
def upload():
  if request.method == 'POST':
    if 'flowchart' not in request.files:
      return("No file part.")

    file = request.files['flowchart']
    if file.filename == '':
      return("No selected file.")
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(os.getcwd(), UPLOAD_FOLDER, filename))
      return("Correct")
    else:
      return "File is not image."

@app.route("/")
def root():
  return render_template('index.html')

if __name__ == "__main__":
  app.run(host="0.0.0.0", debug=True)