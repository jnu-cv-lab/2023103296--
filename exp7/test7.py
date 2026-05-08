import cv2
import numpy as np
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接图片路径
img1_path = os.path.join(script_dir, "box.png")
img2_path = os.path.join(script_dir, "box_in_scene.png")

# 读取图片
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("❌ 图片读取失败！检查文件是否存在")
    exit()

# ---------------------- 任务1：ORB特征检测 ----------------------
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 可视化特征点并保存
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
cv2.imwrite(os.path.join(script_dir, "result1_box_kp.png"), img1_kp)
cv2.imwrite(os.path.join(script_dir, "result1_scene_kp.png"), img2_kp)

print("===== 任务1 结果 =====")
print(f"box 关键点数量：{len(kp1)}")
print(f"scene 关键点数量：{len(kp2)}")
print(f"描述子维度：{des1.shape[1]}")

# ---------------------- 任务2：ORB匹配 ----------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 保存匹配结果
img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
cv2.imwrite(os.path.join(script_dir, "result2_matches.png"), img_match)

print(f"\n===== 任务2 结果 =====")
print(f"总匹配数：{len(matches)}")

# ---------------------- 任务3：RANSAC + 单应矩阵 ----------------------
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()
inliers = sum(matches_mask)

# 保存RANSAC内点结果
img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                             matchesMask=matches_mask, flags=2)
cv2.imwrite(os.path.join(script_dir, "result3_ransac.png"), img_ransac)

print(f"\n===== 任务3 结果 =====")
print(f"单应矩阵 H：\n{H}")
print(f"RANSAC内点数：{inliers}")
print(f"内点比例：{inliers / len(matches):.2f}")

# ---------------------- 任务4：目标定位 ----------------------
h, w = img1.shape
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, H)

img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
img2_res = cv2.polylines(img2_color, [np.int32(dst)], True, (0,0,255), 3)
cv2.imwrite(os.path.join(script_dir, "result4_localization.png"), img2_res)

print(f"\n===== 任务4 结果 =====")
print("✅ 所有结果已保存到exp7文件夹！")
print("生成的文件：")
print("- result1_box_kp.png / result1_scene_kp.png")
print("- result2_matches.png")
print("- result3_ransac.png")
print("- result4_localization.png")