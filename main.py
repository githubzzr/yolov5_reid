import base64
import threading
import webbrowser
from threading import Thread
import os
import shutil
import base64
import torch
from flask_socketio import SocketIO, emit
from flask import *
import cv2
from ultralytics import YOLO
from PIL import Image
import io
from moviepy.editor import VideoFileClip
from flask_cors import CORS, cross_origin
from person_search.yolov8sample import detect
# import logging
# # make your awesome app
# logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
socketio = SocketIO(app)  # 初始化SocketIO
camera = cv2.VideoCapture(0)  # 0通常是默认摄像头







camera_event = threading.Event()

def gen_frames():
    model = YOLO('yolov8n.pt')
    camera = cv2.VideoCapture(0)  # 0通常是默认摄像头
    while not camera_event.isSet():
        success, frame = camera.read()  # 读取一帧视频
        if not success:
            print("error")
            break
        else:
            results = model.predict(frame, classes=0)

            ret, buffer = cv2.imencode('.jpg', results[0].plot())
            frame = buffer.tobytes()
            frame_base64 = base64.b64encode(frame).decode('utf-8')
            socketio.emit('frame', {'data': frame_base64})  # 通过WebSocket发送视频帧
    camera.release()
def increment_filename(filename, directory,clipFlag = False):
    base_name, extension = os.path.splitext(filename)
    counter = 1

    if clipFlag:
        base_name=base_name+"-clip"
    new_filename = filename

    # 检查文件是否存在，并为存在的文件名添加后缀
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base_name}({counter}){extension}"
        counter += 1
    return new_filename


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@socketio.on('startCamera')
def video_frame():
    print(1)
    camera_event.clear()
    thread = Thread(target=gen_frames)
    thread.daemon = True  # 设置为守护线程，这样当主程序退出时线程也会被杀死
    thread.start()  # 开始线程

@socketio.on('stopCamera')
def stop_camera():
    print(2)
    camera_event.set()  # 设置停止事件，使gen_frames线程中断循环

@app.route('/page')
def page():
    # 假设 your_page.html 位于 static 目录
    return render_template('page.html')

@app.route('/capModalPage')
def capModalPage():
    selected_video = request.args.get('selectVideo', "ReidVideo1.mp4")
    # 将获取的参数传递到模板中
    return render_template('capModalPage.html', selected_video=selected_video)


@app.route('/queryDashboard')
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

@app.route('/videoDashboard')
def videoDashboard():
    train_directory = './uploadVideo'
    play_directory = './templates/static/playVideo'
    play_files = os.listdir(play_directory)
    train_files = os.listdir(train_directory)


    # 将图片信息传递给前端
    return render_template('videoDashboard.html', playList=play_files,trainList=train_files)

@app.route('/delete-video', methods=['POST'])
def deleteVdieo():
    data = request.json
    filename = data['filename']
    list_type = data['listType']  #

    if list_type == "train":
        directory = './uploadVideo'
    else:
        directory = './templates/static/playVideo'

    file_path = os.path.join(directory ,filename  )
    print(file_path)
    os.remove(file_path)
    return jsonify({"status": "success"}), 200

@app.route('/test')
def test():

    # 将获取的参数传递到模板中
    return render_template('test.html')

@app.route('/uploadvideo', methods=['POST'])
def upload_video():
    print("Form data:", request.form)
    print("files data:", request.files)
    if 'videoFile' not in  request.files:
        return jsonify({'error': 'No file part'}), 400
    file =request.files['videoFile']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension not in ['mp4', 'mov', 'avi']:
            # 如果文件不是主流视频格式，发送一个alert
            return jsonify({'error': 'File is not a supported video format'}), 400

        # 保存文件到服务器的某个位置
        filepath = './uploadVideo/' + file.filename
        file.save(filepath)
        return jsonify({'message': '视频上传成功'}), 200

@app.route('/list-videos', methods=['GET'])
def list_videos():
   print("list")
   train_directory = './uploadVideo'
   play_directory='./templates/static/playVideo'
   query_directory='./queryImg'

   try:
        # 确保目录存在
        play_files= os.listdir( play_directory)
        train_files = os.listdir(train_directory)
        query_files=os.listdir(query_directory)

        # 可以选择过滤非视频文件，如果需要
        return jsonify({'trainList': train_files, 'playList': play_files,"queryList":query_files}), 200
   except FileNotFoundError:
        return jsonify({'error': 'Directory not found'}), 404


@app.route('/trainImg', methods=['POST'])
def train_img():
    if 'imgFile' not in request.files:
        return 'No file part', 400
    file = request.files['imgFile']
    print(file)
    print(file.filename)
    if file.filename == '':
        return 'No selected file', 400
        # 使用Pillow读取图片
    basePath = './templates/static/image/'
    image = Image.open(file.stream)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # 不管图片原来是什么格式，都转换为JPEG并保存
    with io.BytesIO() as output:
        # 将图片保存为JPEG格式到内存中的output对象
        image.save(output, format='JPEG')
        output.seek(0)  # 将指针移回开始位置以便从头读取数据

        # 定义要保存的文件名，这里假设你想根据原始文件名保存，但扩展名改为.jpg
        output_filename = file.filename.rsplit('.', 1)[0] + '.jpg'
        path=basePath+output_filename

        # 将内存中的JPEG数据写入磁盘文件
        with open(path, 'wb') as f:
            f.write(output.read())



    print(path)
    model = YOLO('yolov8n.pt')
    results = model.predict(source=path  , classes=0, show=False, conf=0.3)
    ret, buffer = cv2.imencode('.jpg', results[0].plot())
    if ret:
        # 成功编码，将字节序列buffer作为响应体返回
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=True,
            attachment_filename='output.jpg'
        )
    else:
        print("Failed to encode image")
        return "Failed to encode image", 400






@app.route('/trainVideo', methods=['POST'])
def train_video():

    selected_file = request.json.get('VideoName')
    frameRate = int(request.json.get('frameRate', 1) or 1)
    print(f"frameRate is{ frameRate}")
    print(selected_file)
    path='./uploadVideo/' + selected_file

    model = YOLO('yolov8n.pt')


    cap = cv2.VideoCapture(path)
    # 获取输入视频的帧率和尺寸信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    baseFile='./templates/static/playVideo/'
    videoFile=baseFile+'temp.mp4'
    # 循环遍历视频帧
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(videoFile, fourcc, fps, (width, height))

    # 加载YOLOv8模型并进行识别
    # 这里需要你使用相应的YOLOv8代码库进行模型加载和识别
    frame_count = 0  # 初始化帧计数器


    try:
            while cap.isOpened():
                # 读取输入视频的一个视频帧
                success, frame = cap.read()

                if not success:
                    break

                if frame_count  % frameRate == 0:
                    results = model.predict(source=frame, classes=0, show=False, conf=0.3)
                    annotated_frame = results[0].plot()

                else:
                    annotated_frame = frame  # 在不进行预测的帧中，直接使用原始帧



                # 写入帧数据到输出视频文件中
                output_video.write(annotated_frame)
                frame_count =frame_count+1

    except Exception as e:
            print(f"处理视频时发生错误: {e}")
            return jsonify(f"处理视频时发生错误: {e}"), 400

    finally:
                cap.release()
                output_video.release()
            # try:
                selected_file = increment_filename(selected_file, baseFile)
                print(videoFile)
                print(baseFile + selected_file)
                clip = VideoFileClip(videoFile)
                # 导出为快速启动模式的 MP4 文件
                clip.write_videofile(os.path.join(baseFile,selected_file), codec="libx264", preset="fast",
                                     ffmpeg_params=["-movflags", "faststart"])
                os.remove(videoFile)  # 删除临时文件
            # except Exception as e:
                # print(f"处理视频后清理资源时发生错误: {e}")
                # return jsonify(f"处理视频后清理资源时发生错误: {e}") ,400

    return jsonify({'message': f'文件: {selected_file} 训练完成'}), 200


@app.route('/trainVideoWithQuery', methods=['POST'])
def trainVideoWithQuery():
    videoName = request.form['VideoName']

    queryName= request.form.get('queryName')
    clipFlag = request.form.get("clipFlag", "true").lower() == "true"
    print(clipFlag)




    current_dir = os.path.dirname(os.path.abspath(__file__))

    print(queryName)
    videoInputFile =os.path.join(current_dir , "uploadVideo/" +videoName)

    videoOutputFile=os.path.join(current_dir , "outPutVideo/" +videoName)
    imgPath = os.path.join(current_dir , "queryImg/" +queryName)
    print(imgPath)
    filename, file_extension = os.path.splitext(queryName)
    queryLabel= filename
    with torch.no_grad():
       startTime,endTime = detect(img_path=imgPath,videoInputFile=videoInputFile,videoOutputFile=videoOutputFile,labelWord=queryLabel,dist_thres=1)

    clip = VideoFileClip(videoOutputFile)
    print(clipFlag)
    if clipFlag:
        if (endTime - startTime > 2):
             clip = clip.subclip(startTime, endTime)




    # 导出为快速启动模式的 MP4 文件
    baseFile = './templates/static/playVideo/'
    videoName = increment_filename( videoName, baseFile,clipFlag)
    clip.write_videofile(os.path.join(baseFile,videoName) , codec="libx264", preset="fast",
                         ffmpeg_params=["-movflags", "faststart"])
    if (endTime - startTime > 2):
        message=f"找到目标，位于原视频的第{startTime}秒到第{endTime}秒"
    else:
        message="目标不存在于原视频"



    return jsonify({'message': message}), 200






@app.route('/get-video')
@cross_origin()
def get_video():
    videoType=request.args.get("videoType","play")
    if (videoType == "play"):
        videoFile = './templates/static/playVideo'
    else:
        videoFile= './uploadVideo'

    video_name = request.args.get('videoName')  # 默认视频文件名
    file_path = videoFile+'/'+video_name
    print(file_path)
    return send_file(file_path, as_attachment=True, mimetype='video/mp4')

@app.route('/uploadQuery', methods=['POST'])
def uploadquery():
    if 'queryImg' not in request.files:
        return 'No file part', 400
    queryImg = request.files['queryImg']
    print(queryImg)
    if queryImg.filename == '':
        return 'No selected file', 400
        # 使用Pillow读取图片

    image = Image.open(queryImg.stream)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    queryLabel = request.form.get('queryLabel') or "target"
    # img_filename = queryLabel.rsplit('.', 1)[0] + '.jpg'
    img_filename = queryLabel + '.jpg'
    imgPath = os.path.join(current_dir, "queryImg/" + img_filename)

    # 不管图片原来是什么格式，都转换为JPEG并保存
    with io.BytesIO() as output:
        # 将图片保存为JPEG格式到内存中的output对象
        image.save(output, format='JPEG')
        output.seek(0)  # 将指针移回开始位置以便从头读取数据

        # 定义要保存的文件名，这里假设你想根据原始文件名保存，但扩展名改为.jpg

        # 将内存中的JPEG数据写入磁盘文件
        with open(imgPath, 'wb') as f:
            f.write(output.read())

    return jsonify({'message': "图片上传成功"}), 200

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
    print('http://127.0.0.1:5000/page')
    # threading.Timer(1, lambda: webbrowser.open_new('http://127.0.0.1:5000/page')).start()
    socketio.run(app, debug=False)  # 使用socketio.run来启动应用
    # print('http://127.0.0.1:5000/page')