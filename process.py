import os
import numpy as np
import torch
import torch.nn as nn
from meshsegnet import *
from sklearn import neighbors
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import scipy.io as sio
import shutil
import time
from sklearn.svm import SVC  # uncomment this line if you don't install thudersvm
# from thundersvm import SVC # comment this line if you don't install thudersvm
from sklearn.neighbors import KNeighborsClassifier
from pygco import cut_from_graph
import open3d as o3d
import json


# 复制数据
def clone_runoob(li1):
    li_copy = li1[:]
    return li_copy


# 对离群点重新进行分类
def class_inlier_outlier(label_list, mean_points,cloud, ind, label_index, points, labels):
    label_change = clone_runoob(labels)
    outlier_index = clone_runoob(label_index)
    ind_reverse = clone_runoob(ind)
    # 得到离群点的label下标
    ind_reverse.reverse()
    for i in ind_reverse:
        outlier_index.pop(i)

    # 获取离群点
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_points = np.array(outlier_cloud.points)

    for i in range(len(outlier_points)):
        distance = []
        for j in range(len(mean_points)):
            dis = np.linalg.norm(outlier_points[i] - mean_points[j], ord=2)  # 计算tooth和GT质心之间的距离
            distance.append(dis)
        min_index = distance.index(min(distance))  # 获取和离群点质心最近label的index
        outlier_label = label_list[min_index]  # 获取离群点应该的label
        index = outlier_index[i]
        label_change[index] = outlier_label

    return label_change





# 利用knn算法消除离群点
def remove_outlier(points, labels):
    # points = np.array(point_cloud_o3d_orign.points)
    # global label_list
    same_label_points = {}

    same_label_index = {}

    mean_points = []  # 所有label种类对应点云的质心坐标

    label_list = []
    for i in range(len(labels)):
        label_list.append(labels[i])
    label_list = list(set(label_list))  # 去重获从小到大排序取GT_label=[0, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
    label_list.sort()
    label_list = label_list[1:]

    for i in label_list:
        key = i
        points_list = []
        all_label_index = []
        for j in range(len(labels)):
            if labels[j] == i:
                points_list.append(points[j].tolist())
                all_label_index.append(j)  # 得到label为 i 的点对应的label的下标
        same_label_points[key] = points_list
        same_label_index[key] = all_label_index

        tooth_mean = np.mean(points_list, axis=0)
        mean_points.append(tooth_mean)
        # print(mean_points)

    for i in label_list:
        points_array = same_label_points[i]
        # 建立一个o3d的点云对象
        pcd = o3d.geometry.PointCloud()
        # 使用Vector3dVector方法转换
        pcd.points = o3d.utility.Vector3dVector(points_array)

        # 对label i 对应的点云进行统计离群值去除，找出离群点并显示
        # 统计式离群点移除
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)  # cl是选中的点，ind是选中点index
        # 可视化
        # display_inlier_outlier(pcd, ind)

        # 对分出来的离群点重新分类
        label_index = same_label_index[i]
        labels = class_inlier_outlier(label_list, mean_points, pcd, ind, label_index, points, labels)
        # print(f"label_change{labels[4400]}")

    return labels


# 消除离群点，保存最后的输出
def remove_outlier_main(jaw, pcd_points, labels, instances_labels):
    # point_cloud_o3d_orign = o3d.io.read_point_cloud('E:/tooth/data/MeshSegNet-master/test_upsample_15/upsample_01K17AN8_upper_refined.pcd')
    # 原始点
    points = pcd_points.copy()
    label = remove_outlier(points, labels)

    # 保存json文件
    label_dict = {}
    label_dict["id_patient"] = ""
    label_dict["jaw"] = jaw
    label_dict["labels"] = label.tolist()
    label_dict["instances"] = instances_labels.tolist()
    b = json.dumps(label_dict)
    with open(os.path.join(save_path, 'dental-labels' + '.json'), 'w') as f_obj:
        f_obj.write(b)
    f_obj.close()


same_points_list = {}


# 体素下采样
def voxel_filter(point_cloud, leaf_size):
    same_points_list = {}
    filtered_points = []
    # step1 计算边界点
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)  # 计算 x,y,z三个维度的最值
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)

    # step2 确定体素的尺寸
    size_r = leaf_size

    # step3 计算每个 volex的维度 voxel grid
    Dx = (x_max - x_min) // size_r + 1
    Dy = (y_max - y_min) // size_r + 1
    Dz = (z_max - z_min) // size_r + 1

    # print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))

    # step4 计算每个点在volex grid内每一个维度的值
    h = list()  # h 为保存索引的列表
    for i in range(len(point_cloud)):
        hx = np.floor((point_cloud[i][0] - x_min) // size_r)
        hy = np.floor((point_cloud[i][1] - y_min) // size_r)
        hz = np.floor((point_cloud[i][2] - z_min) // size_r)
        h.append(hx + hy * Dx + hz * Dx * Dy)
    # print(h[60581])

    # step5 对h值进行排序
    h = np.array(h)
    h_indice = np.argsort(h)  # 提取索引,返回h里面的元素按从小到大排序的  索引
    h_sorted = h[h_indice]  # 升序
    count = 0  # 用于维度的累计
    step = 20
    # 将h值相同的点放入到同一个grid中，并进行筛选
    for i in range(1, len(h_sorted)):  # 0-19999个数据点
        # if i == len(h_sorted)-1:
        #     print("aaa")
        if h_sorted[i] == h_sorted[i - 1] and (i != len(h_sorted) - 1):
            continue
        elif h_sorted[i] == h_sorted[i - 1] and (i == len(h_sorted) - 1):
            point_idx = h_indice[count:]
            key = h_sorted[i - 1]
            same_points_list[key] = point_idx
            _G = np.mean(point_cloud[point_idx], axis=0)  # 所有点的重心
            _d = np.linalg.norm(point_cloud[point_idx] - _G, axis=1, ord=2)  # 计算到重心的距离
            _d.sort()
            inx = [j for j in range(0, len(_d), step)]  # 获取指定间隔元素下标
            for j in inx:
                index = point_idx[j]
                filtered_points.append(point_cloud[index])
            count = i
        elif h_sorted[i] != h_sorted[i - 1] and (i == len(h_sorted) - 1):
            point_idx1 = h_indice[count:i]
            key1 = h_sorted[i - 1]
            same_points_list[key1] = point_idx1
            _G = np.mean(point_cloud[point_idx1], axis=0)  # 所有点的重心
            _d = np.linalg.norm(point_cloud[point_idx1] - _G, axis=1, ord=2)  # 计算到重心的距离
            _d.sort()
            inx = [j for j in range(0, len(_d), step)]  # 获取指定间隔元素下标
            for j in inx:
                index = point_idx1[j]
                filtered_points.append(point_cloud[index])

            point_idx2 = h_indice[i:]
            key2 = h_sorted[i]
            same_points_list[key2] = point_idx2
            _G = np.mean(point_cloud[point_idx2], axis=0)  # 所有点的重心
            _d = np.linalg.norm(point_cloud[point_idx2] - _G, axis=1, ord=2)  # 计算到重心的距离
            _d.sort()
            inx = [j for j in range(0, len(_d), step)]  # 获取指定间隔元素下标
            for j in inx:
                index = point_idx2[j]
                filtered_points.append(point_cloud[index])
            count = i

        else:
            point_idx = h_indice[count: i]
            key = h_sorted[i - 1]
            same_points_list[key] = point_idx
            _G = np.mean(point_cloud[point_idx], axis=0)  # 所有点的重心
            _d = np.linalg.norm(point_cloud[point_idx] - _G, axis=1, ord=2)  # 计算到重心的距离
            _d.sort()
            inx = [j for j in range(0, len(_d), step)]  # 获取指定间隔元素下标
            for j in inx:
                index = point_idx[j]
                filtered_points.append(point_cloud[index])
            count = i

    # 把点云格式改成array，并对外返回
    # print(f'filtered_points[0]为{filtered_points[0]}')
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points,same_points_list


# 体素上采样
def voxel_upsample(same_points_list, point_cloud, filtered_points, filter_labels, leaf_size):
    upsample_label = []
    upsample_point = []
    upsample_index = []
    # step1 计算边界点
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)  # 计算 x,y,z三个维度的最值
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    # step2 确定体素的尺寸
    size_r = leaf_size
    # step3 计算每个 volex的维度 voxel grid
    Dx = (x_max - x_min) // size_r + 1
    Dy = (y_max - y_min) // size_r + 1
    Dz = (z_max - z_min) // size_r + 1
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))

    # step4 计算每个点（采样后的点）在volex grid内每一个维度的值
    h = list()
    for i in range(len(filtered_points)):
        hx = np.floor((filtered_points[i][0] - x_min) // size_r)
        hy = np.floor((filtered_points[i][1] - y_min) // size_r)
        hz = np.floor((filtered_points[i][2] - z_min) // size_r)
        h.append(hx + hy * Dx + hz * Dx * Dy)

    # step5 根据h值查询字典same_points_list
    h = np.array(h)
    count = 0
    for i in range(1, len(h)):
        if h[i] == h[i - 1] and i != (len(h) - 1):
            continue
        elif h[i] == h[i - 1] and i == (len(h) - 1):
            label = filter_labels[count:]
            key = h[i - 1]
            count = i
            # 累计label次数，classcount：{‘A’:2,'B':1}
            classcount = {}
            for i in range(len(label)):
                vote = label[i]
                classcount[vote] = classcount.get(vote, 0) + 1
            # 对map的value排序
            sortedclass = sorted(classcount.items(), key=lambda x: (x[1]), reverse=True)
            # key = h[i-1]
            point_index = same_points_list[key]  # h对应的point index列表
            for j in range(len(point_index)):
                upsample_label.append(sortedclass[0][0])
                index = point_index[j]
                upsample_point.append(point_cloud[index])
                upsample_index.append(index)
        elif h[i] != h[i - 1] and (i == len(h) - 1):
            label1 = filter_labels[count:i]
            key1 = h[i - 1]
            label2 = filter_labels[i:]
            key2 = h[i]
            count = i

            classcount = {}
            for i in range(len(label1)):
                vote = label1[i]
                classcount[vote] = classcount.get(vote, 0) + 1
            sortedclass = sorted(classcount.items(), key=lambda x: (x[1]), reverse=True)
            # key1 = h[i-1]
            point_index = same_points_list[key1]
            for j in range(len(point_index)):
                upsample_label.append(sortedclass[0][0])
                index = point_index[j]
                upsample_point.append(point_cloud[index])
                upsample_index.append(index)

            # label2 = filter_labels[i:]
            classcount = {}
            for i in range(len(label2)):
                vote = label2[i]
                classcount[vote] = classcount.get(vote, 0) + 1
            sortedclass = sorted(classcount.items(), key=lambda x: (x[1]), reverse=True)
            # key2 = h[i]
            point_index = same_points_list[key2]
            for j in range(len(point_index)):
                upsample_label.append(sortedclass[0][0])
                index = point_index[j]
                upsample_point.append(point_cloud[index])
                upsample_index.append(index)
        else:
            label = filter_labels[count:i]
            key = h[i - 1]
            count = i
            classcount = {}
            for i in range(len(label)):
                vote = label[i]
                classcount[vote] = classcount.get(vote, 0) + 1
            sortedclass = sorted(classcount.items(), key=lambda x: (x[1]), reverse=True)
            # key = h[i-1]
            point_index = same_points_list[key]  # h对应的point index列表
            for j in range(len(point_index)):
                upsample_label.append(sortedclass[0][0])
                index = point_index[j]
                upsample_point.append(point_cloud[index])
                upsample_index.append(index)
            # count = i

    # 恢复原始顺序
    # print(f'upsample_index[0]的值为{upsample_index[0]}')
    # print(f'upsample_index的总长度为{len(upsample_index)}')

    # 恢复index原始顺序
    upsample_index = np.array(upsample_index)
    upsample_index_indice = np.argsort(upsample_index)  # 提取索引,返回h里面的元素按从小到大排序的  索引
    upsample_index_sorted = upsample_index[upsample_index_indice]

    upsample_point = np.array(upsample_point)
    upsample_label = np.array(upsample_label)
    # 恢复point和label的原始顺序
    upsample_point_sorted = upsample_point[upsample_index_indice]
    upsample_label_sorted = upsample_label[upsample_index_indice]

    return upsample_point_sorted, upsample_label_sorted


# 利用knn算法上采样
def KNN_sklearn_Load_data(voxel_points, center_points, labels):
    # 载入数据
    # x_train, x_test, y_train, y_test = train_test_split(center_points, labels, test_size=0.1)
    # 构建模型
    model = neighbors.KNeighborsClassifier(n_neighbors=3)
    model.fit(center_points, labels)
    prediction = model.predict(voxel_points.reshape(1, -1))
    # meshtopoints_labels = classification_report(voxel_points, prediction)
    return prediction[0]


# 加载点进行knn上采样
def Load_data(voxel_points, center_points, labels):
    meshtopoints_labels = []
    # meshtopoints_labels.append(SVC_sklearn_Load_data(voxel_points[i], center_points, labels))
    for i in range(0, voxel_points.shape[0]):
        meshtopoints_labels.append(KNN_sklearn_Load_data(voxel_points[i], center_points, labels))
    return np.array(meshtopoints_labels)


'''
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    '''
'''
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i+1]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # default 0
'''


# sortedClassCount = sorted(classCount.items(), key=lambda d: d[1], reverse=True)
# return sortedClassCount[0][0]

# 将三角网格数据上采样回原始点云数据
def mesh_to_points_main(jaw, pcd_points, center_points, labels):
    points = pcd_points.copy()
    # 下采样
    voxel_points, same_points_list = voxel_filter(points, 0.6)

    after_labels = Load_data(voxel_points, center_points, labels)

    upsample_point, upsample_label = voxel_upsample(same_points_list, points, voxel_points, after_labels, 0.6)

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(upsample_point)
    instances_labels = upsample_label.copy()
    # '''
    # o3d.io.write_point_cloud(os.path.join(save_path, 'upsample_' + name + '.pcd'), new_pcd, write_ascii=True)
    for i in range(0, upsample_label.shape[0]):
        if jaw == 'upper':
            if (upsample_label[i] >= 1) and (upsample_label[i] <= 8):
                upsample_label[i] = upsample_label[i] + 10
            elif (upsample_label[i] >= 9) and (upsample_label[i] <= 16):
                upsample_label[i] = upsample_label[i] + 12
        else:
            if (upsample_label[i] >= 1) and (upsample_label[i] <= 8):
                upsample_label[i] = upsample_label[i] + 30
            elif (upsample_label[i] >= 9) and (upsample_label[i] <= 16):
                upsample_label[i] = upsample_label[i] + 32
    remove_outlier_main(jaw, pcd_points, upsample_label, instances_labels)


# 将原始点云数据转换为三角网格
def mesh_grid(pcd_points):
    new_pcd,_ = voxel_filter(pcd_points, 0.6)
    # pcd需要有法向量

    # estimate radius for rolling ball
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(new_pcd)
    pcd_new.estimate_normals()
    distances = pcd_new.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 6 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_new,
        o3d.utility.DoubleVector([radius, radius * 2]))
    # o3d.io.write_triangle_mesh("./tooth date/test.ply", mesh)

    return mesh


# 读取obj文件内容
def read_obj(obj_path):
    with open(obj_path) as file:
        points = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            elif strs[0] == "f":
                faces.append((int(strs[1]), int(strs[2]), int(strs[3])))
            elif strs[1][0:5] == 'lower':
                jaw = 'lower'
            elif strs[1][0:5] == 'upper':
                jaw = 'upper'

    points = np.array(points)
    faces = np.array(faces)

    return points, faces, jaw


# obj文件转为pcd文件
def obj2pcd(obj_path):
    if os.path.exists(obj_path):
        print('yes')
    points, _, jaw = read_obj(obj_path)
    pcd_list = []
    num_points = np.shape(points)[0]
    for i in range(num_points):
        new_line = str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
        pcd_list.append(new_line.split())

    pcd_points = np.array(pcd_list).astype(np.float64)
    return pcd_points, jaw


if __name__ == '__main__':
    obj_path = "/opt/algorithm/input/3d-teeth-scan.obj"
    # ground_truth_path = './tooth_ground-truth_labels/0132CR0A/0132CR0A_lower.json'
    save_path = '/opt/algorithm/output/'
    # gpu_id = utils.get_avail_gpu()
    # gpu_id = 0
    # torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

    # upsampling_method = 'SVM'
    upsampling_method = 'KNN'

    model_path = '/opt/algorithm/Mesh_Segementation_MeshSegNet_17_classes_60samples_best.tar'

    # mesh_path = './data'  # need to modify
    # sample_filenames = ['0132CR0A_upper.pcd'] # need to modify

    # ground_truth_path = './tooth_ground-truth_labels'

    # output_path = './outputs_15'
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)

    num_classes = 17
    num_channels = 15

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

    # load trained model
    # checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Predicting
    model.eval()
    with torch.no_grad():
        pcd_points, jaw = obj2pcd(obj_path)
        mesh = mesh_grid(pcd_points)

        # move mesh to origin
        print('\tPredicting...')

        vertices_points = np.asarray(mesh.vertices)
        triangles_points = np.asarray(mesh.triangles)
        N = triangles_points.shape[0]
        cells = np.zeros((triangles_points.shape[0], 9))
        cells = vertices_points[triangles_points].reshape(triangles_points.shape[0], 9)

        mean_cell_centers = mesh.get_center()
        cells[:, 0:3] -= mean_cell_centers[0:3]
        cells[:, 3:6] -= mean_cell_centers[0:3]
        cells[:, 6:9] -= mean_cell_centers[0:3]

        v1 = np.zeros([triangles_points.shape[0], 3], dtype='float32')
        v2 = np.zeros([triangles_points.shape[0], 3], dtype='float32')
        v1[:, 0] = cells[:, 0] - cells[:, 3]
        v1[:, 1] = cells[:, 1] - cells[:, 4]
        v1[:, 2] = cells[:, 2] - cells[:, 5]
        v2[:, 0] = cells[:, 3] - cells[:, 6]
        v2[:, 1] = cells[:, 4] - cells[:, 7]
        v2[:, 2] = cells[:, 5] - cells[:, 8]
        mesh_normals = np.cross(v1, v2)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]

        # preprae input
        # points = mesh.points().copy()
        points = vertices_points.copy()
        points[:, 0:3] -= mean_cell_centers[0:3]
        normals = np.nan_to_num(mesh_normals).copy()
        barycenters = np.zeros((triangles_points.shape[0], 3))
        s = np.sum(vertices_points[triangles_points], 1)
        barycenters = 1 / 3 * s
        center_points = barycenters.copy()
        # np.save(os.path.join(output_path, name + '.npy'), barycenters)
        barycenters -= mean_cell_centers[0:3]

        # normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))

        # computing A_S and A_L
        A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
        A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
        D = distance_matrix(X[:, 9:12], X[:, 9:12])
        A_S[D < 0.1] = 1.0
        A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

        A_L[D < 0.2] = 1.0
        A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

        # numpy -> torch.tensor
        X = X.transpose(1, 0)
        X = X.reshape([1, X.shape[0], X.shape[1]])
        X = torch.from_numpy(X).to(device, dtype=torch.float)
        A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
        A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
        A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
        A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

        tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
        patch_prob_output = tensor_prob_output.cpu().numpy()

        # refinement
        print('\tRefining by pygco...')
        round_factor = 100
        patch_prob_output[patch_prob_output < 1.0e-6] = 1.0e-6

        # unaries
        unaries = -round_factor * np.log10(patch_prob_output)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, num_classes)

        # parawise
        pairwise = (1 - np.eye(num_classes, dtype=np.int32))

        cells = cells.copy()

        cell_ids = np.asarray(triangles_points)

        lambda_c = 20
        edges = np.empty([1, 3], order='C')
        for i_node in range(cells.shape[0]):
            # Find neighbors
            nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
            nei_id = np.where(nei == 2)
            for i_nei in nei_id[0][:]:
                if i_node < i_nei:
                    cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]) / np.linalg.norm(
                        normals[i_node, 0:3]) / np.linalg.norm(normals[i_nei, 0:3])
                    if cos_theta >= 1.0:
                        cos_theta = 0.9999
                    theta = np.arccos(cos_theta)
                    phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                    if theta > np.pi / 2.0:
                        edges = np.concatenate(
                            (edges, np.array([i_node, i_nei, -np.log10(theta / np.pi) * phi]).reshape(1, 3)), axis=0)
                    else:
                        beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                        edges = np.concatenate(
                            (edges, np.array([i_node, i_nei, -beta * np.log10(theta / np.pi) * phi]).reshape(1, 3)),
                            axis=0)
        edges = np.delete(edges, 0, 0)
        edges[:, 2] *= lambda_c * round_factor
        edges = edges.astype(np.int32)

        refine_labels = cut_from_graph(edges, unaries, pairwise)
        refine_labels = refine_labels.reshape([-1, 1])

        predicted_labels_3 = refine_labels.reshape(refine_labels.shape[0])
        mesh_to_points_main(jaw, pcd_points, center_points, predicted_labels_3)
        # direction["labels"] = (predicted_labels_3.astype(np.int)).tolist()
        # json_str = json.dumps(direction)
        # with open(os.path.join(output_path, name + '_refined.json'), 'w') as f_obj:
        #     f_obj.write(json_str)
