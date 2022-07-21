import torch
import numpy as np
import torch.nn.functional as F

def weighting_DSC(y_pred, y_true, class_weights, smooth=1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1e-7
    mdsc = 0.0
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c] / class_weights.sum()
        mdsc += w * ((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))

    return mdsc


def weighting_SEN(y_pred, y_true, class_weights, smooth=1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1e-7
    msen = 0.0
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c] / class_weights.sum()
        msen += w * ((intersection + smooth) / (true_flat.sum() + smooth))

    return msen


def weighting_PPV(y_pred, y_true, class_weights, smooth=1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1e-7
    mppv = 0.0
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c] / class_weights.sum()
        mppv += w * ((intersection + smooth) / (pred_flat.sum() + smooth))

    return mppv


def points_distance(points, labels, n_classes):
    label_list = [n for n in range(0, n_classes)]
    # for i in range(len(GT_labels)):
    #     label_list.append(GT_labels[i])
    # label_list = list(set(label_list))  # 去重获从小到大排序取GT_label=[0, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
    # label_list.sort(reverse=False)

    # label_list = [38,47,48,0]

    tooth_dic = {}
    tooth_distance = {}

    # 分别获取GT的points,label文件
    for i in label_list:  # 逐个取出label中的元素
        key = i
        tooth_list = []
        distance = []
        for k in range(len(labels)):
            if labels[k] == i:
                tooth_list.append(points[k])
        if tooth_list != []:
            mean = np.mean(tooth_list, axis=0)
            tooth_dic[key] = mean
            for point in tooth_list:
                distance.append(np.linalg.norm(point - mean, ord=2))
            distance.sort(reverse=True)
            tooth_distance[key] = distance[0]
        else:
            tooth_distance[key] = 0  # 如果为缺失牙，不存在它的labels，就补为0

    return tooth_distance


def distance_count(points, y_pre_max):
    r = 0.2
    L_smooth = 0.
    #distance = 0.
    device = points.device
    one_array_points = torch.ones(points.shape).to(device)
    one_array_pred = torch.ones(y_pre_max.values.shape).to(device)
    for i in range(0, points.shape[0]):
        distance_array = torch.linalg.norm((points - one_array_points * points[i]), ord=2, axis=1, keepdims=True)
        # torch.le(distance_array, r)
        distance_sort, idx = torch.sort(distance_array)
        distance_array_pred = torch.le(distance_array, distance_sort[15])
        n = torch.count_nonzero(distance_sort[0:15])
        L_smooth += torch.linalg.norm((y_pre_max.values.reshape(distance_array_pred.shape) * distance_array_pred - y_pre_max.values[i] * one_array_pred.reshape(distance_array_pred.shape) * distance_array_pred), ord=2)/n

    return L_smooth/points.shape[0]



'''
def distance_count(points, y_pre_max):
    r = 0.2
    L_smooth = 0.
    count = 0
    for i in points:
        distance = []
        for j in points:
            if(i == j).all():
                continue
            else:
                distance.append(np.linalg.norm((j - i), ord=2))
        distance_id = sorted(range(len(distance)), key=lambda k:distance[k], reverse=False)
        n = 0
        points_id = []
        for id1 in distance_id:
            if distance[id1] <= r and n <= 14:
                points_id.append(id1)
                n = n + 1
            else:
                break
        for id2 in points_id:
            L_smooth += (y_pre_max[id2] - y_pre_max[count])**2
        count = count + 1

    return L_smooth/points.shape[0]
'''

def smooth_Loss_2(ypred, ytrue):
    epsilon = 1e-6
    smooth = 1e-6
    device = ypred.device
    ypred = torch.clamp(ypred, epsilon, 1 - epsilon)
    center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    pairwise_weights_list = [
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]),
        torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]]),
        torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])]
    ## pairwise loss for each col/row MIL
    pairwise_loss = []
    for w in pairwise_weights_list:
        weights = center_weight - w
        weights = weights.view(1, 1, 3, 3).to(device)
        aff_map = F.conv2d(ypred, weights, padding=1)
        cur_loss = aff_map ** 2
        cur_loss = torch.sum(cur_loss * ytrue, dim=(0, 1, 2)) / (torch.sum(ytrue + smooth, dim=(0, 1, 2)))
        pairwise_loss.append(cur_loss)
    losses = torch.mean(torch.stack(pairwise_loss))
    return losses

def smooth_Loss(y_pred, inputs):
    batch_size = y_pred.shape[0]
    L_smooth = 0.
    for number_patient in range(0, batch_size):
        y_pre_max = torch.max(y_pred[number_patient, :], 1)
        points = inputs[number_patient, 9:12, :].T
        L_smooth += distance_count(points, y_pre_max)
    return L_smooth


def Generalized_Dice_Loss(y_pred, y_true, class_weights, inputs, smooth=1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1e-7
    Lambda_size = 0.1
    Lambda_smooth = 2.5
    weight_smooth = 1000
    size_loss = 0.
    loss = 0.
    n_classes = y_pred.shape[-1]
    batch_size = y_pred.shape[0]
    y_pre_np = y_pred.cpu().detach().numpy()
    y_tru_np = y_true.cpu().detach().numpy()
    #smooth_Loss
    L_smooth = smooth_Loss(y_pred,inputs)
    L_smooth = L_smooth * weight_smooth

    # size约束
    L_pre = []
    L_gt = []
    predicted_labels = np.zeros((y_pred.shape[1]))
    true_labels = np.zeros((y_true.shape[1]))

    for number_patient in range(0, batch_size):
        for i_label in range(n_classes):
            predicted_labels[np.argmax(y_pre_np[number_patient, :], axis=-1) == i_label] = i_label
            true_labels[np.argmax(y_tru_np[number_patient, :], axis=-1) == i_label] = i_label

        points = inputs[number_patient, 9:12, :].cpu().detach().numpy().T
        L_pre_dic = points_distance(points, predicted_labels,n_classes)
        L_gt_dic = points_distance(points, true_labels,n_classes)
        for elem in L_gt_dic:
            if elem == 0:
                continue
            else:
                L_pre.append(L_pre_dic[elem])
                L_gt.append(L_gt_dic[elem])

        size_loss += np.linalg.norm(np.array(L_gt) - np.array(L_pre), ord=2)


    size_loss = Lambda_size * size_loss
    # dice损失
    for c in range(0, n_classes):
        pred_flat = y_pred[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()

        # with weight
        w = class_weights[c] / class_weights.sum()
        loss += w * (1 - ((2. * intersection + smooth) /
                          (pred_flat.sum() + true_flat.sum() + smooth)))
    #loss = (loss + size_loss + L_smooth)/3
    #loss = loss * 0.9 + size_loss * 0.1 + L_smooth * 0.2
    all_loss = (loss + size_loss + L_smooth)/2
    return all_loss


def DSC(y_pred, y_true, ignore_background=True, smooth=1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1e-7
    n_classes = y_pred.shape[-1]
    dsc = []
    if ignore_background:
        for c in range(1, n_classes):  # pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))

        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))

        dsc = np.asarray(dsc)

    return dsc


def SEN(y_pred, y_true, ignore_background=True, smooth=1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1e-7
    n_classes = y_pred.shape[-1]
    sen = []
    if ignore_background:
        for c in range(1, n_classes):  # pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))

        sen = np.asarray(sen)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))

        sen = np.asarray(sen)

    return sen


def PPV(y_pred, y_true, ignore_background=True, smooth=1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1e-7
    n_classes = y_pred.shape[-1]
    ppv = []
    if ignore_background:
        for c in range(1, n_classes):  # pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))

        ppv = np.asarray(ppv)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))

        ppv = np.asarray(ppv)

    return ppv
