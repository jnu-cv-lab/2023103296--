import cv2
import numpy as np

# ---------------------------
# 实验5：图像几何变换与透视校正
# ---------------------------

# 1. 生成测试图像
def create_test_image(size=600):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (550, 550), (0, 0, 0), 3)
    cv2.circle(img, (220, 300), 130, (0, 0, 0), 3)
    cv2.rectangle(img, (380, 170), (480, 430), (0, 0, 0), 3)
    for i in range(80, 530, 100):
        cv2.line(img, (i, 80), (i, 520), (100, 100, 100), 2)
        cv2.line(img, (80, i), (520, i), (100, 100, 100), 2)
    return img

src_img = create_test_image()
h, w = src_img.shape[:2]
cv2.imwrite("exp5_original.jpg", src_img)

# 2. 相似变换（旋转+缩放）
M_similar = cv2.getRotationMatrix2D((w//2, h//2), 15, 0.85)
img_similar = cv2.warpAffine(src_img, M_similar, (w, h), borderValue=(255,255,255))

# 3. 仿射变换
pts_src_aff = np.float32([[50, 50], [550, 50], [50, 550]])
pts_dst_aff = np.float32([[80, 100], [520, 60], [60, 530]])
M_affine = cv2.getAffineTransform(pts_src_aff, pts_dst_aff)
img_affine = cv2.warpAffine(src_img, M_affine, (w, h), borderValue=(255,255,255))

# 4. 透视变换
pts_src_per = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts_dst_per = np.float32([[60, 50], [w-50, 40], [40, h-50], [w-40, h-60]])
M_perspective = cv2.getPerspectiveTransform(pts_src_per, pts_dst_per)
img_perspective = cv2.warpPerspective(src_img, M_perspective, (w, h), borderValue=(255,255,255))

# 5. 拼接对比图
compare_img = np.vstack((
    np.hstack((src_img, img_similar)),
    np.hstack((img_affine, img_perspective))
))
cv2.imwrite("exp5_compare.jpg", compare_img)

# 6. A4 透视校正（可选）
def sort_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    sum_pts = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sum_pts)]
    rect[2] = pts[np.argmax(sum_pts)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

import os
if os.path.exists("a4_test.jpg"):
    img = cv2.imread("a4_test.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edge = cv2.Canny(blur, 30, 120)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    vertex = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            vertex = approx.reshape(4, 2).astype(np.float32)
            break

    if vertex is not None:
        vertex = sort_points(vertex)
        a4_w, a4_h = 700, 990
        dst_pts = np.float32([[0, 0], [a4_w, 0], [a4_w, a4_h], [0, a4_h]])
        M_correct = cv2.getPerspectiveTransform(vertex, dst_pts)
        corrected = cv2.warpPerspective(img, M_correct, (a4_w, a4_h))
        cv2.imwrite("exp5_corrected.jpg", corrected)
        print("✅ A4校正完成")

print("✅ 实验5运行完成，所有图片已保存")