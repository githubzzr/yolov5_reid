import base64
import threading
import webbrowser
from threading import Thread
import os
import shutil
from flask_socketio import SocketIO, emit
from flask import *
import cv2
from ultralytics import YOLO
app = Flask(__name__)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
@app.route('/video')
def video():
    # 假设你的视频文件存放在 'path/to/your/video.mp4'
    path_to_video = './templates/static/playVideo/video.mp4'
    # path_to_video = './templates/static/playVideo/video.mp4'
    # path_to_video = './uploadVideo/video.mp4'
    return send_file(path_to_video, as_attachment=True,mimetype='video/mp4')

@app.route('/page')
def page():
    # 假设 your_page.html 位于 static 目录
    return render_template('test.html')
@app.route('/queryDashboard' )
def queryDashboard():
    queryGalleryPath = './queryImg'  # 设置目标文件夹的相对路径
    base_path = os.path.abspath(queryGalleryPath)
    images_info = []  # 用来存储文件路径和Base64编码

    for dirpath, dirnames, filenames in os.walk(base_path):
        for filename in filenames:
            # 将文件名与其目录路径合并，形成绝对路径
            file_path = os.path.join(dirpath, filename)
            # 编码图片为Base64
            image_base64 = encode_image_to_base64(file_path)
            # 将文件名和Base64编码封装成字典
            images_info.append({
                'imageName': filename,
                'image_base64': image_base64
            })

    # 将图片信息传递给前端
    return render_template('queryDashboard.html', images=images_info)
@app.route('/deleteQuery' , methods=['POST'])
def deleteQuery():
     queryGalleryPath = './queryImg'
     queryName= request.json.get('queryName')
     print(queryName)
     file_path = os.path.join(queryGalleryPath, f"{queryName}.jpg")
     os.remove(file_path)
     return jsonify({"status": "success", "message": f"Query '{queryName}' processed successfully."}), 200

@app.route('/updateQuery' , methods=['POST'])
def updateQuery():
    queryGalleryPath = './queryImg'
    newName = request.json.get('newName')
    originalName= request.json.get('originalName')
    print(newName)
    originalFilePath = os.path.join(queryGalleryPath, f"{originalName}.jpg")
    newFilePath = os.path.join(queryGalleryPath, f"{newName}.jpg")
    os.rename(originalFilePath, newFilePath)


    return jsonify({"status": "success", "message": f"Query '{originalName}' processed successfully."}), 200

if __name__ == '__main__':
    print('http://127.0.0.1:5000/queryDashboard')
    app.run(debug=False)