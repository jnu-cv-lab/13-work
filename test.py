import cv2
import numpy as np
import glob
import os

# ===================== 配置参数（和原代码保持一致） =====================
CHECKERBOARD = (9, 6)    # 棋盘内角点 列×行
SQUARE_SIZE = 25.0       # 方格边长 mm
IMAGE_DIR = "/home/jsxl/build/12-work/calib_imgs" # 标定图片存放文件夹
RESULT_DIR = "results"   # 输出角点图、校正图、标定文本

# 校正平衡参数 alpha：0=裁切多无黑边，1=完整保留画面易残留畸变，推荐0.4
UNDIST_ALPHA = 0.4
# 亚像素收敛条件（优化收紧，提升角点精度）
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
# 亚像素搜索窗口放大
SUBPIX_WIN_SIZE = (15, 15)

os.makedirs(RESULT_DIR, exist_ok=True)

# 构建棋盘格世界三维坐标
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 所有图片3D世界坐标
imgpoints = []  # 所有图片2D亚像素角点

# 读取全部图片格式
image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + \
              glob.glob(os.path.join(IMAGE_DIR, "*.png")) + \
              glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))

if len(image_paths) == 0:
    raise ValueError("没有找到图片，请把标定图片放到 images 文件夹中。")
print(f"共读取到 {len(image_paths)} 张图片\n")

gray_shape = None
success_count = 0
single_img_error_list = []  # 存储每张有效图片的重投影误差

# 1. 遍历图片，检测+亚像素优化角点
for idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    if img is None:
        print(f"无法读取图片：{path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        success_count += 1
        objpoints.append(objp)
        # 高精度亚像素角点优化
        corners_subpix = cv2.cornerSubPix(gray, corners, SUBPIX_WIN_SIZE, (-1, -1), SUBPIX_CRITERIA)
        imgpoints.append(corners_subpix)

        # 绘制并保存角点图
        img_draw = img.copy()
        cv2.drawChessboardCorners(img_draw, CHECKERBOARD, corners_subpix, ret)
        save_path = os.path.join(RESULT_DIR, f"corners_{idx + 1}.jpg")
        cv2.imwrite(save_path, img_draw)
        print(f"[成功] {path}，角点结果保存为 {save_path}")
    else:
        print(f"[失败] {path}，未检测到完整棋盘格角点")

print(f"\n成功检测角点的图片数量：{success_count}")
if success_count < 15:
    print("⚠️ 警告：有效标定图片不足15张，标定精度较差，建议补充更多多角度棋盘照片！")
if success_count < 5:
    raise ValueError("有效图片太少，无法完成标定！")

# 2. 相机标定（增加约束，稳定内参与畸变系数）
# CALIB_FIX_ASPECT_RATIO：固定fx/fy比值，减少过拟合
calib_flags = cv2.CALIB_FIX_ASPECT_RATIO
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray_shape,
    None,
    None,
    flags=calib_flags
)

# 3. 逐张计算单张图片重投影误差（方便筛选劣质图片）
total_error = 0.0
for i in range(len(objpoints)):
    imgpoints_projected, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
    )
    err = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
    single_img_error_list.append(err)
    total_error += err
mean_error = total_error / len(objpoints)

# 输出标定核心结果
print("\n========== 标定结果 ==========")
print("相机内参矩阵 K：")
print(camera_matrix)
print("\n畸变参数 D = [k1, k2, p1, p2, k3]：")
print(dist_coeffs.ravel())

print("\n========== 单张图片重投影误差（单位：像素） ==========")
for i, err in enumerate(single_img_error_list):
    print(f"第{i+1}张有效图误差：{err:.4f}")
print(f"\n全局平均重投影误差：{mean_error:.4f} pixel")

# 4. 高精度去畸变模块（核心优化：initUndistortRectifyMap + remap，替代简易undistort）
test_img = cv2.imread(image_paths[0])
h, w = test_img.shape[:2]

# 获取优化内参，alpha控制画面裁切程度
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    alpha=UNDIST_ALPHA,
    newImgSize=(w, h)
)

# 生成像素映射表，逐像素校正，畸变消除效果远优于undistort
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
)
# 双线性插值，校正图像无锯齿、更平滑
undistorted_full = cv2.remap(test_img, map1, map2, interpolation=cv2.INTER_LINEAR)

# 根据ROI裁剪黑边
x, y, w_roi, h_roi = roi
if w_roi > 0 and h_roi > 0:
    undistorted_crop = undistorted_full[y:y + h_roi, x:x + w_roi]
else:
    undistorted_crop = undistorted_full

# 保存各类结果图
cv2.imwrite(os.path.join(RESULT_DIR, "original.jpg"), test_img)
cv2.imwrite(os.path.join(RESULT_DIR, "undistorted_full.jpg"), undistorted_full)
cv2.imwrite(os.path.join(RESULT_DIR, "undistorted_crop.jpg"), undistorted_crop)

# 拼接原图与校正对比图，方便报告使用
compare_canvas = np.hstack([test_img, undistorted_crop])
cv2.imwrite(os.path.join(RESULT_DIR, "compare_origin_undist.jpg"), compare_canvas)

print("\n========== 去畸变图像已保存至 results 文件夹 ==========")
print("1. original.jpg 原始带畸变图像")
print("2. undistorted_full.jpg 完整去畸变图（含边缘黑边）")
print("3. undistorted_crop.jpg 裁剪黑边后的校正图")
print("4. compare_origin_undist.jpg 原图+校正图拼接对比图")

# 5. 保存完整标定文本记录
txt_path = os.path.join(RESULT_DIR, "calibration_result.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("===== 棋盘格标定参数 =====\n")
    f.write(f"内角点尺寸(宽,高)：{CHECKERBOARD}\n")
    f.write(f"方格实际边长：{SQUARE_SIZE} mm\n\n")

    f.write("===== 相机内参矩阵 K =====\n")
    f.write(str(camera_matrix))

    f.write("\n\n===== 畸变系数 D [k1, k2, p1, p2, k3] =====\n")
    f.write(str(dist_coeffs.ravel()))

    f.write("\n\n===== 重投影误差 =====\n")
    f.write(f"全局平均误差：{mean_error:.4f} pixel\n")
    f.write("各单张图片误差列表：\n")
    for idx, err in enumerate(single_img_error_list):
        f.write(f"第{idx+1}张：{err:.4f} px\n")

print(f"\n完整标定参数文件已保存：{txt_path}")