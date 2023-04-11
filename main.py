# This is a sample Python script.
import os.path
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import sys

import maxflow

from MyDialog import Ui_Dialog  # 导入GUI文件
from miis import *  # 嵌入了matplotlib的文件
from PyQt5.QtCore import QPoint
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import pyqtgraph as pg
from scipy import ndimage
from utils import *
import time


class MainDialogImgBW(QDialog, Ui_Dialog):
    def __init__(self):
        super(MainDialogImgBW, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("显示nii图像")
        self.setMinimumSize(0, 0)

        # 创建存放nii文件路径的属性
        self.nii_path = ''
        # 创建存放mask文件路径的属性
        self.mask_path = ''
        # 创建记录nii文件里面图片数量的属性
        self.shape = 1
        # 创建用于检查radio button选择标记的属性，选择'nii图像'，为0，现在‘mask图像’，为1
        self.check = 0

        self.image = None
        self.mask = None

        # 定义MyFigure类的一个实例
        self.F = MyFigure(width=3, height=2, dpi=100)

        # 在GUI的groupBox中创建一个布局，用于添加MyFigure类的实例（即图形）后其他部件。
        self.gridlayout = QGridLayout(self.groupBox)  # 继承容器groupBox
        self.gridlayout.addWidget(self.F, 0, 1)
        self.pushButton.clicked.connect(self.openImage)
        self.pushButton_2.clicked.connect(self.openMask)
        self.pushButton_3.clicked.connect(self.refine)
        self.pushButton_4.clicked.connect(self.saveMask)
        self.horizontalSlider.valueChanged.connect(self.bindSlider)
        self.horizontalSlider.valueChanged.connect(self.bindSpineBox)
        self.spinBox.valueChanged.connect(self.sliderChange)
        self.radioButton.clicked.connect(self.bindradiobutton)
        self.radioButton_2.clicked.connect(self.bindradiobutton)

        self.F.figure.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.fgSeed = []
        self.bgSeed = []
        self.z = 0
        self.spacing = None

    def on_mouse_click(self, event):
        if event.inaxes:
            if event.button == 1:
                x, y = event.xdata, event.ydata
                self.fgSeed.append([int(self.z), int(x), int(y)])
                # self.label.setText(str(self.curPosition))
                self.showimage(self.z)
            if event.button == 3:
                x, y = event.xdata, event.ydata
                self.bgSeed.append([int(self.z), int(x), int(y)])
                self.showimage(self.z)

    def showimage(self, slice_idx):
        self.z = slice_idx
        # image_nii = sitk.ReadImage(Path(self.nii_path))
        # image = sitk.GetArrayFromImage(image_nii)
        # image = np.asarray(image, np.float32)
        # rawSpacing = image_nii.GetSpacing()
        # self.spacing = [rawSpacing[2], rawSpacing[1], rawSpacing[0]]
        image = self.image
        self.horizontalSlider.setRange(0, image.shape[0] - 1)
        # if not self.mask_path == '':
        #     # data_mask = nib.load(Path(self.mask_path))
        #     # data2 = data_mask.get_fdata()
        #     # data2 = np.rot90(data2)
        #     mask_nii = sitk.ReadImage(Path(self.mask_path))
        #     mask = sitk.GetArrayFromImage(mask_nii)
        #     mask = np.asarray(mask, np.float32)
        #     mask = np.flip(mask)
        #     self.mask = mask
        if self.mask is not None:
            mask = self.mask
        else:
            mask = np.zeros_like(image)
            self.mask = mask

        fig = self.F.figure
        fig.clear()
        ax = fig.add_subplot(111)  # 将画布划成1*1的大小并将图像放在1号位置，给画布加上一个坐标轴
        ax.imshow(image[slice_idx, :, :], cmap='gray')
        # 将mask的矩阵转换，未勾画区为透明的，勾画区为红色
        if self.check == 1:
            array1 = list(mask[slice_idx, :, :])
            a = len(array1)
            b = len(array1[0])
            pic = [[0] * b for i in range(a)]
            for i in range(0, a):
                for j in range(0, b):
                    if array1[i][j] == 0:
                        pic[i][j] = [0, 0, 0, 0]
                    else:
                        pic[i][j] = [255, 0, 0, 100]

            ax.imshow(pic, cmap='viridis')

            for fg in self.fgSeed:
                if fg[0] == slice_idx:
                    ax.plot(fg[1], fg[2], 'o', markersize=3, color='green')

            for bg in self.bgSeed:
                if bg[0] == slice_idx:
                    ax.plot(bg[1], bg[2], 'o', markersize=3, color='blue')

            del array1
            del pic

        fig.canvas.draw()

    def bindradiobutton(self):
        if self.radioButton.isChecked():
            self.check = 0
        else:
            self.check = 1
        slice_idx = self.horizontalSlider.value()
        self.showimage(slice_idx)

    def bindSlider(self):
        slice_idx = self.horizontalSlider.value()
        self.showimage(slice_idx)

    def bindSpineBox(self):
        self.spinBox.setValue(self.horizontalSlider.value())

    def sliderChange(self):
        self.horizontalSlider.setValue(self.spinBox.value())

    def openImage(self):
        self.fgSeed.clear()
        self.bgSeed.clear()
        # self.mask = None
        file_name = QFileDialog.getOpenFileName(None, "Open File", "./", "data(*.nii.gz;*.nii;*.dcm)")
        self.nii_path = file_name[0]
        slice_idx = self.horizontalSlider.value()
        if file_name[0].split('.')[1] == 'nii':
            image_nii = sitk.ReadImage(Path(self.nii_path))
        else:
            reader = sitk.ImageSeriesReader()
            file_name = os.path.dirname(file_name[0])
            img_name = reader.GetGDCMSeriesFileNames(file_name)
            reader.SetFileNames(img_name)
            image_nii = reader.Execute()
        image = sitk.GetArrayFromImage(image_nii)
        self.image = np.asarray(image, np.float32)
        rawSpacing = image_nii.GetSpacing()
        self.spacing = [rawSpacing[2], rawSpacing[1], rawSpacing[0]]
        # self.image = np.flip(self.image)
        self.showimage(slice_idx)

    def openMask(self):
        file_name = QFileDialog.getOpenFileName(None, "Open File", "./", "data(*.nii.gz;*.nii;*.dcm)")
        self.mask_path = file_name[0]
        mask_nii = sitk.ReadImage(Path(self.mask_path))
        mask = sitk.GetArrayFromImage(mask_nii)
        mask = np.asarray(mask, np.float32)
        # mask = np.flip(mask)
        self.mask = mask

    def simple_refine(self):
        fore_seeds = np.zeros_like(self.image)
        for i1 in self.fgSeed:
            fore_seeds[i1[0], i1[1], i1[2]] = 1
        back_seeds = np.zeros_like(self.image)
        for i2 in self.bgSeed:
            back_seeds[i2[0], i2[1], i2[2]] = 1

        fore_seeds = extend_points(fore_seeds)
        back_seeds = extend_points(back_seeds)
        all_refined_seeds = np.maximum(fore_seeds, back_seeds)

        bbox = get_bbox(self.mask)
        bbox = update_bbox(bbox, all_refined_seeds)
        cropped_img = crop_image(self.image, bbox)
        init_seg = [self.mask, 1.0 - self.mask]
        fg_prob = init_seg[0]
        bg_prob = init_seg[1]

        cropped_fore_seg = crop_image(fg_prob, bbox)
        cropped_back_seg = crop_image(bg_prob, bbox)
        cropped_fore_seeds = crop_image(fore_seeds, bbox)
        cropped_back_seeds = crop_image(back_seeds, bbox)

        Prob = np.asarray([cropped_back_seg, cropped_fore_seg])
        Prob = np.transpose(Prob, [1, 2, 3, 0])

        crf_seeds = np.zeros_like(cropped_fore_seeds, np.uint8)
        crf_seeds[cropped_fore_seeds > 0] = 170
        crf_seeds[cropped_back_seeds > 0] = 255
        crf_param = (10.0, 15.0)
        Seed = np.asarray([crf_seeds == 255, crf_seeds == 170], np.uint8)
        Seed = np.transpose(Seed, [1, 2, 3, 0])

        refine_pred = maxflow.interactive_maxflow3d(cropped_img, Prob, Seed, crf_param)
        pred = np.zeros_like(self.image, dtype=np.float32)
        pred[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = refine_pred

        strt = ndimage.generate_binary_structure(3, 1)
        seg = np.asarray(ndimage.binary_opening(pred, strt), np.uint8)
        seg = np.asarray(ndimage.binary_closing(seg, strt), np.uint8)
        seg = connected_component(seg)
        seg = ndimage.binary_fill_holes(seg)

        seg = np.clip(seg, 0, 255)
        seg = np.asarray(seg, np.uint8)

        self.mask = seg
        self.fgSeed.clear()
        self.bgSeed.clear()
        self.showimage(self.z)

    def refine(self):
        t00 = time.time()
        fore_seeds = np.zeros_like(self.image)
        for i1 in self.fgSeed:
            fore_seeds[i1[0], i1[1], i1[2]] = 1
        back_seeds = np.zeros_like(self.image)
        for i2 in self.bgSeed:
            back_seeds[i2[0], i2[1], i2[2]] = 1

        t30 = time.time()
        # fore_seeds = extend_points(fore_seeds)
        # back_seeds = extend_points(back_seeds)
        fore_seeds = extend_points2(fore_seeds)
        back_seeds = extend_points2(back_seeds)
        t31 = time.time()
        print("runtime extend points: {0:}".format(t31 - t30))

        all_refined_seeds = np.maximum(fore_seeds, back_seeds)

        bbox = get_bbox(self.mask)
        bbox = update_bbox(bbox, all_refined_seeds)
        cropped_img = crop_image(self.image, bbox)

        normal_img = itensity_standardization(cropped_img)
        init_seg = [self.mask, 1.0-self.mask]
        fg_prob = init_seg[0]
        bg_prob = init_seg[1]

        cropped_init_seg = crop_image(fg_prob, bbox)
        cropped_back_seg = crop_image(bg_prob, bbox)
        cropped_fore_seeds = crop_image(fore_seeds, bbox)
        cropped_back_seeds = crop_image(back_seeds, bbox)

        spacing = self.spacing
        # spacing = None
        t10 = time.time()
        cropped_fore_geos = interaction_geodesic_distance(normal_img, cropped_fore_seeds, spacing, refine=True)
        cropped_back_geos = interaction_geodesic_distance(normal_img, cropped_back_seeds, spacing, refine=True)
        t11 = time.time()
        print("runtime geodesic distance: {0:}".format(t11 - t10))

        fore_prob = np.maximum(cropped_fore_geos, cropped_init_seg)
        back_prob = np.maximum(cropped_back_geos, cropped_back_seg)

        crf_seeds = np.zeros_like(cropped_fore_seeds, np.uint8)
        crf_seeds[cropped_fore_seeds > 0] = 170
        crf_seeds[cropped_back_seeds > 0] = 255
        crf_param = (20.0, 0.1)
        crf_seeds = np.asarray([crf_seeds == 255, crf_seeds == 170], np.uint8)
        crf_seeds = np.transpose(crf_seeds, [1, 2, 3, 0])

        z, x, y = fore_prob.shape
        prob_feature = np.zeros((2, z, x, y), dtype=np.float32)
        prob_feature[0] = fore_prob
        prob_feature[1] = back_prob
        softmax_feature = np.exp(prob_feature) / np.sum(np.exp(prob_feature), axis=0)
        fg_prob = softmax_feature[0].astype(np.float32)
        bg_prob = softmax_feature[1].astype(np.float32)

        prob = np.asarray([bg_prob, fg_prob])
        prob = np.transpose(prob, [1, 2, 3, 0])

        t20 = time.time()
        refine_pred = maxflow.interactive_maxflow3d(
            normal_img, prob, crf_seeds, crf_param
        )
        t21 = time.time()
        print("runtime maxflow3d: {0:}".format(t21 - t20))

        pred = np.zeros_like(self.image, dtype=np.float32)
        pred[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = refine_pred

        strt = ndimage.generate_binary_structure(3, 3)
        seg = np.asarray(ndimage.binary_opening(pred, strt), np.uint8)
        seg = np.asarray(ndimage.binary_closing(seg, strt), np.uint8)
        seg = connected_component(seg)
        seg = ndimage.binary_fill_holes(seg)

        seg = np.clip(seg, 0, 255)
        seg = np.asarray(seg, np.uint8)

        self.mask = seg
        self.fgSeed.clear()
        self.bgSeed.clear()
        t01 = time.time()
        print("runtime: {0:}".format(t01 - t00))
        self.showimage(self.z)


    def saveMask(self):
        file_name, tp = QFileDialog.getSaveFileName(None, "Save File", "./", "nii(*.nii.gz;*.nii)")
        mask = np.flip(self.mask)
        new_mask_nii = sitk.GetImageFromArray(mask)
        sitk.WriteImage(new_mask_nii, file_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainDialogImgBW()
    main.show()
    sys.exit(app.exec_())