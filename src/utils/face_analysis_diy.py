# coding: utf-8

"""
face detectoin and alignment using InsightFace
"""

import numpy as np
from .rprint import rlog as log
from .dependencies.insightface.app import FaceAnalysis
from .dependencies.insightface.app.common import Face
from .timer import Timer


def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
    if len(faces) <= 0:
        return faces

    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]), reverse=True)
    if direction == 'distance-from-retarget-face':
        return sorted(faces, key=lambda face: (((face['bbox'][2]+face['bbox'][0])/2-face_center[0])**2+((face['bbox'][3]+face['bbox'][1])/2-face_center[1])**2)**0.5)
    return faces


class FaceAnalysisDIY(FaceAnalysis):
    def __init__(self, name='buffalo_l', root='~/.insightface', allowed_modules=None, **kwargs):
        super().__init__(name=name, root=root, allowed_modules=allowed_modules, **kwargs)

        self.timer = Timer()

    def get(self, img_bgr, **kwargs):
        """
        从图像中检测人脸并进行分析。

        :param img_bgr: BGR 格式的输入图像。
        :param kwargs: 可选参数，包括：
            max_face_num: 检测的最大人脸数量，0 表示不限制。
            flag_do_landmark_2d_106: 是否进行 106 点关键点检测。
            direction: 检测结果排序的方向。
        :return: 包含检测和分析结果的列表。
        """
        # 解析可选参数
        max_num = kwargs.get('max_face_num', 0)  # 检测人脸的数量上限，0 表示无限制
        flag_do_landmark_2d_106 = kwargs.get('flag_do_landmark_2d_106', True)  # 是否进行 106 点关键点检测
        direction = kwargs.get('direction', 'large-small')  # 排序方向，如从大到小或从小到大

        # 使用检测模型检测人脸
        bboxes, kpss = self.det_model.detect(img_bgr, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:  # 如果没有检测到人脸
            return []  # 直接返回空列表

        # 初始化结果列表
        ret = []
        # 遍历检测到的每个人脸
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]  # 获取人脸边界框
            det_score = bboxes[i, 4]  # 获取检测得分
            kps = None
            if kpss is not None:  # 如果有关键点信息
                kps = kpss[i]  # 获取关键点坐标

            # 创建人脸对象
            face = Face(bbox=bbox, kps=kps, det_score=det_score)

            # 遍历模型字典，除了检测模型外，对每个模型执行任务
            for taskname, model in self.models.items():
                if taskname == 'detection':  # 如果是检测模型，跳过
                    continue

                if (not flag_do_landmark_2d_106) and taskname == 'landmark_2d_106':  # 如果不需要 106 点关键点检测
                    continue

                # 执行模型的 get 方法，对人脸进行分析
                model.get(img_bgr, face)

            # 将分析后的人脸添加到结果列表
            ret.append(face)

        # 根据方向对结果进行排序
        ret = sort_by_direction(ret, direction, face_center=None)
        return ret  # 返回排序后的人脸分析结果

    def warmup(self):
        self.timer.tic()

        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.get(img_bgr)

        elapse = self.timer.toc()
        log(f'FaceAnalysisDIY warmup time: {elapse:.3f}s')
