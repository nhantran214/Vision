#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, render_template, Markup, redirect
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
from processes import recognize, update_member_to_csv, addnewmember
from face_recognition import load_image_file
import shutil
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)
# Dropzone settings
basedir = os.path.abspath(os.path.dirname(__file__))
# app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'tempFacePhotos')
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'tempFacePhotos'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=30,
)
dropzone = Dropzone(app)
# route http posts to this method
@app.route('/api/recognizeface', methods=['POST'])
def recognizeface():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print ('server print: ', img.shape)
    faces_info = recognize(img)
    return jsonify(faces_info)

@app.route('/test', methods=['POST', 'GET'])
def test():
    if request.method == 'POST':
        if 'pic' in request.files:
            file = request.files['pic']
            filename = secure_filename(file.filename)
            im_path = os.path.join('static', filename)
            file.save(im_path)
            im = load_image_file(im_path)
            # print (im)
            name = recognize(im)
            print (name)
        return render_template('test.html', im_path = im_path, face_name = name)
    else:
        return render_template('test.html')


@app.route('/', methods=['POST', 'GET'])
def listofpeople():
    info = pd.read_csv('FaceDB.csv')
    if request.method == 'GET':
        return render_template('index.html', infoTable = Markup(info.to_html()))


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
    return redirect('/')

@app.route('/addmember', methods=['POST'])
def addmember():
    if request.method == 'POST':
        info = pd.read_csv('FaceDB.csv')
        print (request.form)
        nickname = request.form.get('nickname')
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        job = request.form.get('job')
        if nickname not in info.nickname.tolist():
            info = update_member_to_csv(nickname, name, age, gender, job)
        else:
            comment = "nickname đã tồn tại, xin vui lòng chọn nickname khác "
            return render_template('index.html', infoTable=Markup(info.to_html()), comment=comment)
        print (name,age,gender,job)
        addnewmember(nickname)
    return redirect('/')

# start flask app
server_ip = '192.168.0.113'
app.run(host=server_ip, port=8080, debug=True)
