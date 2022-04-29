import cv2
from cv2 import VideoCapture
from cv2 import imwrite

video_path = "./datasets/culane/1.MP4"  # 视频路径
out_pathmovie = "./datasets/culane/img_"  # 保存图片路径+名字
out_pathlive ="./datasets/culane"
name="\picture"
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    imwrite(address, image)
def mp4tojpg():
    print(video_path)
    videoCapture = VideoCapture(video_path)
    # 读帧
    success, frame = videoCapture.read()
    print(success)
    i=10000
    while success :
        i=i+1
        name = str(i)
        save_image(frame, out_pathmovie, name)
        # 这个操作就是读取一帧，所以要一直循环，不然会只有第一帧
        success, frame = videoCapture.read()
def cameraget():
    #获取摄像头视频,0是内置，1是外部
    videoCapture = cv2.VideoCapture(1)
    return videoCapture