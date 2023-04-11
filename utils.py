import GeodisTK
import numpy as np
import random
import SimpleITK as sitk
from scipy.ndimage import *
from skimage import measure


def max_min_normalize(img):
    """
    Normalize image to 0 - 1
    """
    imgn = (img - img.min()) / max((img.max() - img.min()), 1e-8)
    return imgn


def itensity_standardization(image):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    volume = image[image > 0]
    mean = volume.mean()
    std = volume.std()
    out = (image - mean)/std
    out = out.astype(np.float32)
    return out


def extreme_points(mask, pert=0):
    """
    Extract extreme points from mask.
    """
    def find_point(id_x, id_y, id_z, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id], id_z[sel_id]]

    # List of coordinates of the mask
    inds_x, inds_y, inds_z = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, inds_z, np.where(inds_x <= np.min(inds_x) + pert)),  # min_x
                     find_point(inds_x, inds_y, inds_z, np.where(inds_x >= np.max(inds_x) - pert)),  # max_x
                     find_point(inds_x, inds_y, inds_z, np.where(inds_y <= np.min(inds_y) + pert)),  # min_y
                     find_point(inds_x, inds_y, inds_z, np.where(inds_y >= np.max(inds_y) - pert)),  # max_y
                     find_point(inds_x, inds_y, inds_z, np.where(inds_z <= np.min(inds_z) + pert)),  # min_z
                     find_point(inds_x, inds_y, inds_z, np.where(inds_z >= np.max(inds_z) - pert)),  # max_z
                     ])


def get_bbox(mask, points=None, pad=0, zero_pad=False):
    if points is None:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if zero_pad:
        z_min_bound = -np.inf
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        z_max_bound = np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        z_min_bound = 0
        x_min_bound = 0
        y_min_bound = 0
        z_max_bound = mask.shape[0] - 1
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[2] - 1

    z_min = max(inds[0].min() - pad, z_min_bound)
    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[2].min() - pad, y_min_bound)
    z_max = min(inds[0].max() + pad, z_max_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[2].max() + pad, y_max_bound)

    return [z_min, z_max, x_min, x_max, y_min, y_max]


def update_bbox(bbox, seeds):
    points = np.where(seeds != 0)
    if bbox is not None:
        z_min = min(int(np.min(points[0])), bbox[0])
        z_max = max(int(np.max(points[0])), bbox[1])
        x_min = min(int(np.min(points[1])), bbox[2])
        x_max = max(int(np.max(points[1])), bbox[3])
        y_min = min(int(np.min(points[2])), bbox[4])
        y_max = max(int(np.max(points[2])), bbox[5])
    else:
        z_min = int(np.min(points[0]))
        z_max = int(np.max(points[0]))
        x_min = int(np.min(points[1]))
        x_max = int(np.max(points[1]))
        y_min = int(np.min(points[2]))
        y_max = int(np.max(points[2]))

    return [z_min - 1, z_max + 1, x_min - 1, x_max + 1, y_min - 1, y_max + 1]


def crop_image(image, bbox):
    """
    crop the image according to bbox.
    """
    cropped_img = image[
                  bbox[0]: bbox[1],
                  bbox[2]: bbox[3],
                  bbox[4]: bbox[5]
                  ]
    return cropped_img


def zoom_img(img, size):
    """
    reshape data to size for training and testing
    """
    d, h, w = img.shape
    zoomed_img = zoom(img, (size[0] / d, size[1] / h, size[2] / w))
    return zoomed_img


def extend_points(seed):
    """
    """
    if(seed.sum() > 0):
        points = distance_transform_edt(seed == 0)
        points[points == 0] = 1
        points[points > 2] = 0
        points[points > 0] = 1
    else:
        points = seed
    return points.astype(np.uint8)


def extend_points2(seed):
    points = binary_dilation(seed)
    return points.astype(np.uint8)


def interaction_geodesic_distance(img, seed, spacing=None, threshold=0, refine=False):
    if seed.sum() > 0:
        I = np.asanyarray(img, np.float32)
        S = seed
        # geo_dis = GeodisTK.geodesic3d_fast_marching(I, S, spacing)
        geo_dis = GeodisTK.geodesic3d_raster_scan(I, S, spacing, 1.0, 4)
        if threshold > 0:
            geo_dis[geo_dis > threshold] = threshold
            geo_dis = geo_dis / threshold
        elif not refine:
            geo_dis = np.exp(-geo_dis)
        else:
            geo_dis = np.exp(-geo_dis**2)
    else:
        geo_dis = np.zeros_like(img, dtype=np.float32)
    return max_min_normalize(geo_dis)


def connected_component(image):
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image

    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num + 1)]
    area_list = [region[i - 1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x - 1])[::-1]
    # 去除面积较小的连通域
    if len(num_list_sorted) > 1:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[1:]:
            # label[label==i] = 0
            label[region[i - 1].slice][region[i - 1].image] = 0
    return label

