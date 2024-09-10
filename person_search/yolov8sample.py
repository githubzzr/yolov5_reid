import math

import torch
import torch.nn.functional as F
import os
import sys

from ultralytics import YOLO

sys.path.append('../..')
# from reid.data.transforms import build_transforms

from person_search.tool import *
from person_search.reid.modeling import build_model
from person_search.reid.config import cfg as reidCfg
import numpy as np
from PIL import Image
import cv2


#
# def resize_frame(frame, new_size=(640, 640)):
#     # Convert the NumPy array frame (from OpenCV) to a PIL Image
#     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#     # Define the transformation - resize to new_size
#     transform = transforms.Compose([
#         transforms.Resize(new_size),
#         transforms.ToTensor()  # 转换为Tensor并自动调整通道顺序
#     ])
#
#     # 应用转换
#     transformed_tensor = transform(pil_image)
#     return transformed_tensor

def detect(dist_thres=0.8,labelWord="target",img_path=r"./query/test.jpg",videoInputFile = r"./source/ReidVideo1.mp4",\
           videoOutputFile=r"./source/b.mp4"  ):


    current_dir = os.path.dirname(os.path.abspath(__file__))


    modelWeightFile = os.path.join(current_dir , "weights\ReID_resnet50_ibn_a.pth")


    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False  # set False for reproducible results


    # ---------- 行人重识别模型初始化 --------------------------
    # query_loader, num_query = make_data_loader(reidCfg)  # 验证集预处理

    reidModel = build_model(reidCfg, num_classes=1501)  # 模型初始化
    reidModel.load_param(modelWeightFile)  # 加载权重
    reidModel.to(device).eval()  # 设置为推理模式

    query_feats = []  # 测试特征

    #
    # for i, batch in enumerate(query_loader):
    #     with torch.no_grad():
    #         img, pid, camid = batch  # 返回图片，ID，相机ID
    #         img = img.to(device)  # 将图片放入gpu
    #         print("Shape of query:", img.shape)  # 打印形状
    #         print(type(img))
    #         feat = reidModel(img)  # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
    #         query_feats.append(feat)  # 获得特征值列表


    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))

    try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    except Exception as e:
            print("Error occurred when reading '{}': {}".format(img_path, e))
    transform = build_transforms(reidCfg)
    # print("第一次类型" )
    # print(type(img))

    img=Image.fromarray(np.uint8(img)) #由 numpy 数组表示的图像转换为一个 PIL 图像对象
    # print("第二次类型")
    # print(type(img))

    img = transform(img)
    img = img.unsqueeze(0).to(device)  # 添加一个维度以匹配batch维度
    print("Shape of query:", img.shape)  # 打印形状
    print(type(img))
    feat = reidModel(img)  #
    query_feats.append(feat)  # 获得特征值列表




    query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
    print("The query feature is normalized")
    query_feats = F.normalize(query_feats, dim=1, p=2)  # 计算出查询图片的特征向量 Lp归一化

    # --------------- yolov8 行人检测模型初始化 -------------------
    model = YOLO('yolov8n.pt')
    stride, names, pt = 1, model.names, True

    # Dataloader
    # bs = 1  # batch_size

    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference

    cap = cv2.VideoCapture(videoInputFile)
    # 获取输入视频的帧率和尺寸信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # imgsz = check_img_size((height, width ), s=32)  # check image size
    firstDetectFlag=True
    startTime=0
    endTime=0





    output_video = cv2.VideoWriter(videoOutputFile, fourcc, fps, (width, height))

    try:
       while cap.isOpened():
        # 读取输入视频的一个视频帧
            success, frame = cap.read()

            if not success:
                 break
            # 逐帧或按设定的帧数读取视频的图片
            im=frame
            # print(im.shape)
        #     im = letterbox(frame, new_shape=imgsz, stride=stride, auto=pt)[0]  # padded resize
        # #改成transformer
        #     print("Shape of im:", im.shape)  # 打印形状
           # im = im.transpose((2, 0, 1))  # HWC to CHW
        #     im = im[::-1, :, :]  # 正确的BGR到RGB转换应该在通道维度上进行逆序
        #        im = im.transpose((2, 0, 1))  # HWC to CHW
        #     # 查看转换后的数组的形状和大小
        #     print("Shape of im:", im.shape)  # 打印形状
        #     im = resize_frame(im, imgsz)
        #
        #     print("Shape of im:", im.size)  # 打印形状
        #
        #     im = np.ascontiguousarray(im)  # contiguou
        #     im = torch.from_numpy(im).to(model.device)  # numpy to tensor
        #     im =  im.float()  # uint8 to fp16/32
        #     im /= 255  # 0 - 255 to 0.0 - 1.0 归一化
        #     if len(im.shape) == 3:
        #         im = im[None]  # expand for batch dim 就等于im.unsqueeze(0)

        # Inference


            results = model.predict(source=im, classes=0,show=False)

        # NMS

            # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            result = results[0]
            # result.show()
            # print(result.boxes)\
            gallery_img = []
            gallery_loc = []  # 这个列表用来存放框的坐标
            for box in result.boxes:
              if len(box):
                # Rescale boxes from img_size to im0 size
                confidence = float(box.conf.cpu())
                if confidence < 0.3:
                    continue

                    # 这里是左下角坐标和右上角坐标
                xmin, ymin, xmax, ymax = np.array(box.xyxy.cpu(), dtype=np.int).squeeze()
                width = xmax - xmin
                height = ymax - ymin
                if width * height > 100:  #去除太小的目标图片
                    gallery_loc.append((xmin, ymin, xmax, ymax))
                    crop_img = frame[ymin:ymax,xmin:xmax]
                    # HWC  这个im0是读取的帧，获取该帧中框的位置 im0= <class 'numpy.ndarray'>
                    crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                    crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                    gallery_img.append(crop_img)




            if gallery_img:
                    gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
                    gallery_img = gallery_img.to(device)
                    gallery_feats = reidModel(gallery_img)  # torch.Size([7, 2048])
                    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量

                    m, n = query_feats.shape[0], gallery_feats.shape[0]
                    distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                              torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                    # 首先计算查询集和图库集中每个特征向量的平方和，然后使用广播机制计算所有成对的欧氏距离的平方。
                    # distmat 的每个元素 (i, j) 包含了第 i 个查询特征与第 j 个图库特征之间的距离的平方。
                    distmat.addmm_(1, -2, query_feats, gallery_feats.t())
                    # 使用矩阵乘法计算查询特征与图库特征的点积的两倍，然后从距离平方矩阵中减去这个结果，得到最终的距离矩阵。
                    distmat = distmat.cpu().numpy()
                    distmat = distmat.sum(axis=0) / len(query_feats)  # 平均一下query中同一行人的多个结果

                    index = distmat.argmin()
                    # 最小值的索引

                    if distmat[index] < dist_thres:
                        # print('距离：%s' % distmat[index])
                        #这里可以指定颜色
                        if  firstDetectFlag:
                            firstDetectFlag=False
                            startTime = cap.get(cv2.CAP_PROP_POS_MSEC)
                        endTime= cap.get(cv2.CAP_PROP_POS_MSEC)
                        confidence = 1 / (1 + distmat[index])
                        plot_one_box(gallery_loc[index], frame, label=labelWord,color=(0,0,255),score=confidence)


            # print('one frame done')
            output_video .write(frame)
    finally:
        cap.release()
        output_video.release()
    return startTime/1000,endTime/1000







if __name__ == "__main__":

    with torch.no_grad():
        detect()
