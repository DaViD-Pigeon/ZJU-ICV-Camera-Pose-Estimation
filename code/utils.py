import numpy as np
import matplotlib.pyplot as plt
from hloc.visualization import read_image
import cv2
import os
from skimage.metrics import structural_similarity as ssim



def project_cloud_to_image(model, camera, image_dir, query, ret):
    qvec = ret['qvec'] # 旋转向量
    tvec = ret['tvec'] # 平移向量
    intrinsic = camera.calibration_matrix()
    extrinsic = np.eye(4)
    # 计算旋转矩阵 R
    theta = 2 * np.arccos(qvec[0])  # 角度
    axis = qvec[1:] / np.sin(theta/2)  # 轴
    R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

    # 构建外参矩阵 P
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = tvec

    # print("外参矩阵 P:")
    # print(extrinsic)
    # print(intrinsic)

    # 所有特征点的点云坐标
    # inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    # 所有点云的坐标
    inl_3d_ids = [pid for pid in model.points3D]
    inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(inl_3d_ids)])
    # 将世界坐标系下的点云转换到相机坐标系下
    inl_3d = np.dot(inl_3d, R.T) + tvec
    # 将相机坐标系下的点云转换到像素坐标系下
    inl_2d = np.dot(inl_3d, intrinsic.T)
    # 从齐次坐标转换到非齐次坐标
    inl_2d = inl_2d[:, :2] / inl_2d[:, 2:]

    # print(inl_2d.shape)
    # 所有特征点的颜色
    # inl_3d_color = np.array([model.points3D[pid].color for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    # 所有点云的颜色
    inl_3d_color = np.array([model.points3D[pid].color for pid in np.array(inl_3d_ids)])

    width = camera.width
    height = camera.height
    # print(width, height)
    # 剔除超出图像范围的点云
    inl_3d_color = inl_3d_color[(inl_2d[:, 0] >= 0) & (inl_2d[:, 0] < width) & (inl_2d[:, 1] >= 0) & (inl_2d[:, 1] < height)]
    inl_2d = inl_2d[(inl_2d[:, 0] >= 0) & (inl_2d[:, 0] < width) & (inl_2d[:, 1] >= 0) & (inl_2d[:, 1] < height)]

    # inl_2d_img = np.zeros((width, height, 3))

    # inl_2d_img[inl_2d[:, 0].astype(int), inl_2d[:, 1].astype(int)] = inl_3d_color

    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    # plt.imshow(inl_2d_img)
    plt.scatter(inl_2d[:, 0], inl_2d[:, 1], c=inl_3d_color / 255.0, s=1)
    # 设置y轴的方向
    plt.gca().invert_yaxis()

    plt.imshow(np.zeros((width, height, 3)))
    plt.subplot(1, 2, 2)
    plt.imshow(read_image(image_dir / query))
    plt.show()

def save_cloud_to_image(model, camera, ret, save_path):
    qvec = ret['qvec'] # 旋转向量
    tvec = ret['tvec'] # 平移向量
    intrinsic = camera.calibration_matrix()
    extrinsic = np.eye(4)
    # 计算旋转矩阵 R
    theta = 2 * np.arccos(qvec[0])  # 角度
    axis = qvec[1:] / np.sin(theta/2)  # 轴
    R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

    # 构建外参矩阵 P
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = tvec

    # print("外参矩阵 P:")
    # print(extrinsic)
    # print(intrinsic)

    # 所有特征点的点云坐标
    # inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    # 所有点云的坐标
    inl_3d_ids = [pid for pid in model.points3D]
    inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(inl_3d_ids)])
    # 将世界坐标系下的点云转换到相机坐标系下
    inl_3d = np.dot(inl_3d, R.T) + tvec
    # 将相机坐标系下的点云转换到像素坐标系下
    inl_2d = np.dot(inl_3d, intrinsic.T)
    # 从齐次坐标转换到非齐次坐标
    inl_2d = inl_2d[:, :2] / inl_2d[:, 2:]

    # print(inl_2d.shape)
    # 所有特征点的颜色
    # inl_3d_color = np.array([model.points3D[pid].color for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    # 所有点云的颜色
    inl_3d_color = np.array([model.points3D[pid].color for pid in np.array(inl_3d_ids)])

    width = camera.width
    height = camera.height
    # print(width, height)
    # 剔除超出图像范围的点云
    inl_3d_color = inl_3d_color[(inl_2d[:, 0] >= 0) & (inl_2d[:, 0] < width) & (inl_2d[:, 1] >= 0) & (inl_2d[:, 1] < height)]
    inl_2d = inl_2d[(inl_2d[:, 0] >= 0) & (inl_2d[:, 0] < width) & (inl_2d[:, 1] >= 0) & (inl_2d[:, 1] < height)]

    # inl_2d_img = np.zeros((width, height, 3))

    # inl_2d_img[inl_2d[:, 0].astype(int), inl_2d[:, 1].astype(int)] = inl_3d_color

    
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(inl_2d_img)
    plt.scatter(inl_2d[:, 0], inl_2d[:, 1], c=inl_3d_color / 255.0, s=1)
    # 设置y轴的方向
    plt.gca().invert_yaxis()

    plt.imshow(np.zeros((width, height, 3)))
    plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(read_image(image_dir / query))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print("save image to {}".format(save_path))
    plt.show()

def project_feature_cloud_to_image(model, camera, image_dir, query, ret, log):
    qvec = ret['qvec'] # 旋转向量
    tvec = ret['tvec'] # 平移向量
    intrinsic = camera.calibration_matrix()
    extrinsic = np.eye(4)
    # 计算旋转矩阵 R
    theta = 2 * np.arccos(qvec[0])  # 角度
    axis = qvec[1:] / np.sin(theta/2)  # 轴
    R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

    # 构建外参矩阵 P
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = tvec

    # print("外参矩阵 P:")
    # print(extrinsic)
    # print(intrinsic)

    # 所有特征点的点云坐标
    inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    # 所有点云的坐标
    # inl_3d_ids = [pid for pid in model.points3D]
    # inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(inl_3d_ids)])
    # 将世界坐标系下的点云转换到相机坐标系下
    inl_3d = np.dot(inl_3d, R.T) + tvec
    # 将相机坐标系下的点云转换到像素坐标系下
    inl_2d = np.dot(inl_3d, intrinsic.T)
    # 从齐次坐标转换到非齐次坐标
    inl_2d = inl_2d[:, :2] / inl_2d[:, 2:]

    # print(inl_2d.shape)
    # 所有特征点的颜色
    inl_3d_color = np.array([model.points3D[pid].color for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    # 所有点云的颜色
    # inl_3d_color = np.array([model.points3D[pid].color for pid in np.array(inl_3d_ids)])

    width = camera.width
    height = camera.height
    # print(width, height)
    # 剔除超出图像范围的点云
    inl_3d_color = inl_3d_color[(inl_2d[:, 0] >= 0) & (inl_2d[:, 0] < width) & (inl_2d[:, 1] >= 0) & (inl_2d[:, 1] < height)]
    inl_2d = inl_2d[(inl_2d[:, 0] >= 0) & (inl_2d[:, 0] < width) & (inl_2d[:, 1] >= 0) & (inl_2d[:, 1] < height)]

    # inl_2d_img = np.zeros((width, height, 3))

    # inl_2d_img[inl_2d[:, 0].astype(int), inl_2d[:, 1].astype(int)] = inl_3d_color

    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    # plt.imshow(inl_2d_img)
    plt.scatter(inl_2d[:, 0], inl_2d[:, 1], c=inl_3d_color / 255.0, s=1)
    # 设置y轴的方向
    plt.gca().invert_yaxis()

    plt.imshow(np.zeros((width, height, 3)))
    plt.subplot(1, 2, 2)
    plt.imshow(read_image(image_dir / query))
    plt.show()

def save_feature_cloud_to_image(model, camera, ret, log, save_path):
    qvec = ret['qvec'] # 旋转向量
    tvec = ret['tvec'] # 平移向量
    intrinsic = camera.calibration_matrix()
    extrinsic = np.eye(4)
    # 计算旋转矩阵 R
    theta = 2 * np.arccos(qvec[0])  # 角度
    axis = qvec[1:] / np.sin(theta/2)  # 轴
    R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

    # 构建外参矩阵 P
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = tvec

    # print("外参矩阵 P:")
    # print(extrinsic)
    # print(intrinsic)

    # 所有特征点的点云坐标
    inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])])
    # 所有点云的坐标
    # inl_3d_ids = [pid for pid in model.points3D]
    # inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(inl_3d_ids)])
    # 将世界坐标系下的点云转换到相机坐标系下
    inl_3d = np.dot(inl_3d, R.T) + tvec
    # 将相机坐标系下的点云转换到像素坐标系下
    inl_2d = np.dot(inl_3d, intrinsic.T)
    # 从齐次坐标转换到非齐次坐标
    inl_2d = inl_2d[:, :2] / inl_2d[:, 2:]

    # print(inl_2d.shape)
    # 所有特征点的颜色
    inl_3d_color = np.array([model.points3D[pid].color for pid in np.array(log['points3D_ids'])])
    # 所有点云的颜色
    # inl_3d_color = np.array([model.points3D[pid].color for pid in np.array(inl_3d_ids)])

    width = camera.width
    height = camera.height
    # print(width, height)
    # 剔除超出图像范围的点云
    inl_3d_color = inl_3d_color[(inl_2d[:, 0] >= 0) & (inl_2d[:, 0] < width) & (inl_2d[:, 1] >= 0) & (inl_2d[:, 1] < height)]
    inl_2d = inl_2d[(inl_2d[:, 0] >= 0) & (inl_2d[:, 0] < width) & (inl_2d[:, 1] >= 0) & (inl_2d[:, 1] < height)]

    # inl_2d_img = np.zeros((width, height, 3))

    # inl_2d_img[inl_2d[:, 0].astype(int), inl_2d[:, 1].astype(int)] = inl_3d_color

    
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(inl_2d_img)
    plt.scatter(inl_2d[:, 0], inl_2d[:, 1], c=inl_3d_color / 255.0, s=1)
    # 设置y轴的方向
    plt.gca().invert_yaxis()

    plt.imshow(np.zeros((width, height, 3)))
    plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(read_image(image_dir / query))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print("save image to {}".format(save_path))
    plt.show()

def save_origin_features(image_dir, query, ret, log, camera, save_path):
    width = camera.width
    height = camera.height
    mkp_q = log['keypoints_query'][ret['inliers']]
    print(mkp_q.shape)
    # 从原图中提取特征点的颜色
    origin_image = read_image(image_dir / query)
    print(origin_image.shape)
    mkp_q_colors = np.array([origin_image[int(mkp_q[i, 1]), int(mkp_q[i, 0])] for i in range(mkp_q.shape[0])])
    plt.scatter(mkp_q[:, 0], mkp_q[:, 1], c = mkp_q_colors / 255.0, s=1)
    # 设置y轴的方向
    plt.gca().invert_yaxis()

    plt.imshow(np.zeros((width, height, 3)))
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print("save image to {}".format(save_path))
    plt.show()

def save_origin_features_within_cloud(image_dir, query, ret, log, camera, save_path):
    width = camera.width
    height = camera.height
    mkp_q = log['keypoints_query']
    print(mkp_q.shape)
    # 从原图中提取特征点的颜色
    origin_image = read_image(image_dir / query)
    print(origin_image.shape)
    mkp_q_colors = np.array([origin_image[int(mkp_q[i, 1]), int(mkp_q[i, 0])] for i in range(mkp_q.shape[0])])
    plt.scatter(mkp_q[:, 0], mkp_q[:, 1], c = mkp_q_colors / 255.0, s=1)
    # 设置y轴的方向
    plt.gca().invert_yaxis()

    plt.imshow(np.zeros((width, height, 3)))
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print("save image to {}".format(save_path))
    plt.show()


#------------------------------------------------------------
# p-hash evaluate
def location_evaluate(src_path, dst_path):
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)
    # 对原图进行处理，若在dst中为黑色，则在src中也为黑色
    src_copy = src.copy()
    # print(src_copy.shape)
    # print(dst.shape)
    src_copy[src_copy != 0] = 255
    dst_copy = dst.copy()
    dst_copy[dst_copy != 0] = 255
    # 利用p-hash算法计算两张图片的相似度
    return hash_similarity(src_copy, dst_copy)
    

def get_img_p_hash(img):
    """
    Get the pHash value of the image, pHash : Perceptual hash algorithm(感知哈希算法)
    :param img: img in MAT format(img = cv2.imread(image))
    :return: pHash value
    """
    hash_len = 128

    # GET Gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize image, use the different way to get the best result
    resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_AREA)

    # Change the int of image to float, for better DCT
    height, width = resize_gray_img.shape[:2]
    vis0 = np.zeros((height, width), np.float32)
    vis0[:height, :width] = resize_gray_img

    # DCT: Discrete cosine transform(离散余弦变换)
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(hash_len, hash_len)
    img_list = vis1.flatten()

    # Calculate the avg value
    avg = sum(img_list) * 1.0 / len(img_list)
    avg_list = []
    for i in img_list:
        temp = '1' if i > avg else '0'
        avg_list.append(temp)
    # Calculate the hash value
    p_hash_str = ''
    for x in range(0, hash_len * hash_len, 4):
        p_hash_str += '%x' % int(''.join(avg_list[x:x + 4]), 2)
    return p_hash_str

def ham_dist(x, y):
    """
    Get the hamming distance of two values.
        hamming distance(汉明距)
    :param x:
    :param y:
    :return: the hamming distance
    """
    return sum([ch1 != ch2 for ch1, ch2 in zip(x, y)])

def hash_similarity(img1, img2):
    """
    Get the similarity of two pictures via pHash
    """
    hash_img1 = get_img_p_hash(img1)
    hash_img2 = get_img_p_hash(img2)
    difference = 0
    assert len(hash_img1) == len(hash_img2)
    for i in range(len(hash_img1)):
        if hash_img1[i] != hash_img2[i]:
            difference += 1
    return 1 - difference / len(hash_img1)

#------------------------------------------------------------
# ssim evaluate
def ssim_evaluate(src_path, dst_path):
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)
    src_copy = src.copy()
    assert src.shape == dst.shape
    src_copy[src_copy != 0] = 255
    dst_copy = dst.copy()
    dst_copy[dst_copy != 0] = 255
    # 利用ssim算法计算两张图片的相似度
    return ssim(src_copy, dst_copy, channel_axis=2)



