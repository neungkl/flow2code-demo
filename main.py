import os
from flask import Flask, render_template, abort, redirect, url_for, request, send_from_directory
from werkzeug.utils import secure_filename
from credential import VISION_API_TOKEN
from model import flow2code
app = Flask(__name__)

UPLOAD_FOLDER = 'upload'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in set(['jpg', 'png', 'jpeg'])

@app.route('/list')
def dir_listing():
    abs_path = os.path.join(os.getcwd(), "upload")

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Show directory contents
    files = os.listdir(abs_path)
    return render_template('files.html', files=files)

@app.route("/upload", methods=['POST'])
def upload():
  if request.method == 'POST':
    if 'flowchart' not in request.files:
      return "No file part."

    file = request.files['flowchart']
    if file.filename == '':
      return "No selected file."
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(os.getcwd(), UPLOAD_FOLDER, filename))
      return redirect(url_for('flow', img_path=filename))
    else:
      return "File is not image."

@app.route("/")
def root():
  abs_path = os.path.join(os.getcwd(), "upload")
  files = os.listdir(abs_path)[0:7]
  return render_template('index.html', files=files)

@app.route("/flow/<img_path>")
def flow(img_path):
  img_full_path = os.path.join(os.getcwd(), UPLOAD_FOLDER, img_path)
  if not os.path.isfile(img_full_path):
    return abort(404)

  result = flow2code(img_full_path)
  return render_template(
    'flow.html',
    tokens=result["tokens"],
    positions=result["positions"].tolist(),
    img_width=result["img_width"],
    img_height=result["img_height"],
    img_path=img_path,
    vision_api_token=VISION_API_TOKEN
  )

@app.route('/img/<path>')
def send_img(path):
  return send_from_directory(os.path.join(os.getcwd(), UPLOAD_FOLDER), path)

if __name__ == "__main__":
  app.run(host="0.0.0.0", debug=True)