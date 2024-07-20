# coding: utf-8

"""
Pipeline of LivePortrait
"""

import torch

from .utils.viz import viz_lmk

torch.backends.cudnn.benchmark = True  # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2;

cv2.setNumThreads(0);
cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_driving_info, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def execute(self, args: ArgumentConfig):
        inf_cfg = self.live_portrait_wrapper.inference_cfg  # 推理配置
        device = self.live_portrait_wrapper.device  # 设备信息
        crop_cfg = self.cropper.crop_cfg  # 裁剪配置

        # 处理源肖像图像
        img_rgb = load_image_rgb(args.source_image)  # 加载RGB格式的源图像
        img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)  # 将图像调整到最大尺寸限制
        log(f"从 {args.source_image} 加载源图像")  # 日志记录加载的源图像路径

        # 使用裁剪配置裁剪源图像
        crop_info = self.cropper.crop_source_image(img_rgb, crop_cfg)
        if crop_info is None:  # 如果没有检测到人脸
            raise Exception("在源图像中未检测到人脸!")  # 抛出异常

        source_lmk = crop_info['lmk_crop']  # 获取裁剪后的人脸关键点
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']  # 裁剪后的图像和固定大小（256x256）的图像
        # 保存裁剪后的图像为 img_crop.jpg
        # 输出日志
        log(f"裁剪后的图像已保存为 img_crop.jpg。")
        cv2.imwrite("img_crop.jpg", img_crop)
        cv2.imwrite("img_crop_256x256.jpg", img_crop_256x256)

        # 如果需要裁剪，准备裁剪后的图像；否则，将原始图像强制缩放到256x256
        if inf_cfg.flag_do_crop:
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        else:
            img_crop_256x256 = cv2.resize(img_rgb, (256, 256))
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)

        # 获取关键点信息
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']  # 关键点位置
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])  # 获取旋转矩阵
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)  # 提取3D特征
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)  # 转换关键点

        # 判断是否需要将嘴唇比例设置为零
        flag_lip_zero = inf_cfg.flag_lip_zero  # 是否将嘴唇张开度设为0
        if flag_lip_zero:
            # 在动画之前让嘴唇张开度为0
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(
                c_d_lip_before_animation, source_lmk)
            lip_zero_threshold = combined_lip_ratio_tensor_before_animation[0][0]  # 计算嘴唇张开度阈值
            if lip_zero_threshold < inf_cfg.lip_zero_threshold:  # 如果低于设定的阈值，则不执行设零操作
                flag_lip_zero = False
            else:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s,
                                                                                     combined_lip_ratio_tensor_before_animation)  # 否则，重新定位嘴唇关键点

        # 处理驱动信息，包括从模板加载或视频文件中提取数据
        flag_load_from_template = is_template(args.driving_info)  # 判断是否从模板加载
        driving_rgb_crop_256x256_lst = None  # 初始化裁剪后的RGB帧列表
        wfp_template = None  # 初始化模板文件路径

        if flag_load_from_template:
            # 如果是从模板加载，则快速加载模板信息，但视频和音频裁剪无效
            log(f"从模板加载：{args.driving_info}，不是视频，所以裁剪的视频和音频都是NULL。", style='bold green')
            template_dct = load(args.driving_info)  # 加载模板字典
            n_frames = template_dct['n_frames']  # 获取帧数

            # 设置输出帧率
            output_fps = template_dct.get('output_fps', inf_cfg.output_fps)
            log(f'模板的FPS：{output_fps}')

            if args.flag_crop_driving_video:
                log("警告：flag_crop_driving_video为真，但驱动信息来自模板，因此该选项被忽略。")

        elif osp.exists(args.driving_info) and is_video(args.driving_info):
            # 如果从视频文件加载，并创建运动模板
            log(f"加载视频：{args.driving_info}")
            if osp.isdir(args.driving_info):
                output_fps = inf_cfg.output_fps
            else:
                output_fps = int(get_fps(args.driving_info))  # 获取视频帧率
                log(f'{args.driving_info}的FPS是：{output_fps}')

            log(f"加载视频文件 (mp4 mov avi等...)：{args.driving_info}")
            driving_rgb_lst = load_driving_info(args.driving_info)  # 加载驱动信息

            # 开始创建运动模板
            log("开始创建运动模板...")
            if inf_cfg.flag_crop_driving_video:
                ret = self.cropper.crop_driving_video(driving_rgb_lst)  # 裁剪视频帧
                log(f'驱动视频已裁剪，处理了{len(ret["frame_crop_lst"])}帧。')
                driving_rgb_crop_lst, driving_lmk_crop_lst = ret['frame_crop_lst'], ret['lmk_crop_lst']  # 分离裁剪后的帧和关键点
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in
                                                driving_rgb_crop_lst]  # 将裁剪后的帧缩放到256x256
            else:
                driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)  # 从裁剪视频计算关键点
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # 强制缩放所有帧到256x256

            c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_driving_ratio(
                driving_lmk_crop_lst)  # 计算眼睛和嘴唇的比例
            # 准备驱动视频
            I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_crop_256x256_lst)
            template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst,
                                                     output_fps=output_fps)  # 创建运动模板

            wfp_template = remove_suffix(args.driving_info) + '.pkl'  # 构建模板文件名
            dump(wfp_template, template_dct)  # 保存模板到文件
            log(f"将运动模板保存到 {wfp_template}")

            n_frames = I_d_lst.shape[0]  # 获取模板帧数
        else:
            raise Exception(f"{args.driving_info}不存在或不支持的驱动信息类型！")

        ######## prepare for pasteback ########
        # 准备用于粘贴回去（pasteback）的帧
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            # 准备粘贴回去所需要的掩码
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'],
                                                dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            I_p_pstbk_lst = []
            log("粘贴回去的掩码准备完成。")
        #########################################

        I_p_lst = []
        R_d_0, x_d_0_info = None, None

        for i in track(range(n_frames), description='🚀Animating...', total=n_frames):
            # 加载每帧的运动信息
            x_d_i_info = template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            R_d_i = x_d_i_info['R_d']

            # 保存第一帧的运动信息
            if i == 0:
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info

            # 根据配置决定是否使用相对运动
            if inf_cfg.flag_relative_motion:
                # 计算新的旋转、delta、scale和t
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                R_new = R_d_i
                delta_new = x_d_i_info['exp']
                scale_new = x_s_info['scale']
                t_new = x_d_i_info['t']

            # 设置tz为0
            t_new[..., 2].fill_(0)
            # 计算新的变形
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            # 根据不同的配置进行不同的操作
            if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # 如果不需要缝合或重定向
                if flag_lip_zero:
                    x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # 如果需要缝合但不需要重定向
                if flag_lip_zero:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s,
                                                                     x_d_i_new) + lip_delta_before_animation.reshape(-1,
                                                                                                                     x_s.shape[
                                                                                                                         1],
                                                                                                                     3)
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            else:
                # 处理眼睛和嘴唇的重定向
                eyes_delta, lip_delta = None, None
                if inf_cfg.flag_eye_retargeting:
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                if inf_cfg.flag_lip_retargeting:
                    lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

                # 更新变形
                x_d_i_new = x_d_i_new + (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + (
                    lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                if inf_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            # 应用变形并解码
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            # 如果需要粘贴回去
            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        # 输出结果
        mkdir(args.output_dir)
        wfp_concat = None
        flag_has_audio = (not flag_load_from_template) and has_audio_stream(args.driving_info)

        # 拼接最终结果
        frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, img_crop_256x256, I_p_lst)
        wfp_concat = osp.join(args.output_dir,
                              f'{basename(args.source_image)}--{basename(args.driving_info)}_concat.mp4')
        images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)

        # 如果有音频流，添加音频
        if flag_has_audio:
            wfp_concat_with_audio = osp.join(args.output_dir,
                                             f'{basename(args.source_image)}--{basename(args.driving_info)}_concat_with_audio.mp4')
            add_audio_to_video(wfp_concat, args.driving_info, wfp_concat_with_audio)
            os.replace(wfp_concat_with_audio, wfp_concat)
            log(f"替换 {wfp_concat} 为 {wfp_concat_with_audio}")

        # 保存动画结果
        wfp = osp.join(args.output_dir, f'{basename(args.source_image)}--{basename(args.driving_info)}.mp4')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            images2video(I_p_pstbk_lst, wfp=wfp, fps=output_fps)
        else:
            images2video(I_p_lst, wfp=wfp, fps=output_fps)

        # 如果有音频流，再次添加音频
        if flag_has_audio:
            wfp_with_audio = osp.join(args.output_dir,
                                      f'{basename(args.source_image)}--{basename(args.driving_info)}_with_audio.mp4')
            add_audio_to_video(wfp, args.driving_info, wfp_with_audio)
            os.replace(wfp_with_audio, wfp)
            log(f"替换 {wfp} 为 {wfp_with_audio}")

        # 最终日志
        log(f'动画模板: {wfp_template}, 下次你可以使用 `-d` 参数指定这个模板路径，避免重新裁剪视频，制作运动模板和保护隐私。')
        log(f'动画视频: {wfp}')
        log(f'拼接动画视频: {wfp_concat}')

        return wfp, wfp_concat

    def make_motion_template(self, I_d_lst, c_d_eyes_lst, c_d_lip_lst, **kwargs):
        n_frames = I_d_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_d_eyes_lst': [],
            'c_d_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s_d, R_d, δ_d and t_d for inference
            I_d_i = I_d_lst[i]
            x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
            R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

            item_dct = {
                'scale': x_d_i_info['scale'].cpu().numpy().astype(np.float32),
                'R_d': R_d_i.cpu().numpy().astype(np.float32),
                'exp': x_d_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_d_i_info['t'].cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_d_eyes = c_d_eyes_lst[i].astype(np.float32)
            template_dct['c_d_eyes_lst'].append(c_d_eyes)

            c_d_lip = c_d_lip_lst[i].astype(np.float32)
            template_dct['c_d_lip_lst'].append(c_d_lip)

        return template_dct
