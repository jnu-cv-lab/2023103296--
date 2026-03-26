import cv2
import numpy as np

# --------------------------
# 任务1: 读取WSL内部的测试图片
# --------------------------
img = cv2.imread("/home/tangzhenzhou/cv-course/exp1/images/test.jpg")
if img is None:
    print("图片读取失败，请检查路径！")
    exit()

# --------------------------
# 任务2: 输出图像基本信息
# --------------------------
height, width, channels = img.shape
dtype = img.dtype
print(f"图像尺寸：宽度={width}，高度={height}")
print(f"通道数：{channels}")
print(f"像素数据类型：{dtype}")

# --------------------------
# 任务3: 显示原图（OpenCV方式）
# --------------------------
cv2.imshow("原图", img)
cv2.waitKey(0)  # 等待按键后关闭窗口
cv2.destroyAllWindows()

# --------------------------
# 任务4: 转换为灰度图并显示
# --------------------------
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("灰度图", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------
# 任务5: 保存灰度图
# --------------------------
cv2.imwrite("gray_test.jpg", img_gray)
print("灰度图已保存为 gray_test.jpg")

# --------------------------
# 任务6: NumPy操作
# --------------------------
pixel_bgr = img[50, 50]
print(f"像素(50,50)的BGR值：{pixel_bgr}")

crop_img = img[0:100, 0:100]
cv2.imwrite("crop_test.jpg", crop_img)
print("左上角裁剪区域已保存为 crop_test.jpg")