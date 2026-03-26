实验一：OpenCV 图像基本操作
 
实验目的
 
1. 掌握在 WSL 环境下配置 Python + OpenCV 开发环境
2. 学会使用 OpenCV 读取、显示、保存图像
3. 了解图像基本属性（尺寸、通道、数据类型）
4. 实现图像灰度化、像素访问与区域裁剪等基础操作
 
实验环境
 
- 系统：WSL Ubuntu
- 语言：Python 3
- 库：opencv-python、numpy
 
实验内容
 
1. 读取本地图片  test.jpg 
2. 输出图像宽度、高度、通道数、像素类型
3. 将彩色图像转换为灰度图像并保存
4. 访问指定坐标像素值
5. 裁剪图像左上角区域并保存
 
实验结果文件
 
-  main.py ：主程序代码
-  images/test.jpg ：原始输入图片
-  gray_test.jpg ：灰度化结果图
-  crop_corner.jpg ：裁剪结果图
 
运行方式
 
bash
  
cd ~/cv-course/exp1
python main.py
