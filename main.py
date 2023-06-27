import os
import imageio

from PIL import Image, ImageSequence
import cv2

def parseGIF(gifname):
    # 将gif解析为图片
    # 读取GIF
    im = Image.open(gifname)
    # GIF图片流的迭代器
    iter = ImageSequence.Iterator(im)
    # 获取文件名
    file_name = gifname.split(".")[0]
    index = 1
    # 遍历图片流的每一帧
    for frame in iter:
        print("image %d: mode %s, size %s" % (index, frame.mode, frame.size))
        frame.save("/data_1/zhongchongyang/ATA/exps/Motion interpolation/frame%d.png" % (index))
        index += 1

def parsevideo(name):
    v = cv2.VideoCapture(name)
    index = 0
    s = True
    while s:
        s, img = v.read()
        cv2.imwrite("/data_1/zhongchongyang/ATA/exps/humanact12_gcntcnfilmin/drink_subfigures_fig_5100/frame%d.png" % (index),img)
        if cv2.waitKey(10) ==27:
            break
        index +=1

if __name__ == "__main__":
    #parseGIF("/data_1/zhongchongyang/ATA/exps/Motion interpolation/fig_5100_interpolation.gif")
    #parsevideo("/data_1/zhongchongyang/ATA/amass_viz/smpl_fig/comparison/conflict.mp4")
    images = []
    for i in range(25):
        im = Image.open('/data_1/zhongchongyang/GAGCN/fig/v2/purchases/69.7375prepurchases frame_{}.png'.format(i+1))
        images.append(im)
    images[0].save("/data_1/zhongchongyang/GAGCN/fig/v2/purchases/69.7375prepurchases frame.gif", save_all=True, loop = True, append_images = images[1:],duration = 0.1)