# ============================================================
#  VGGT-Long：长序列单目3D重建主脚本（详细中文注释版）
#  功能概览：
#    1) 从文件夹加载长序列图像，按“分块+重叠”策略送入 VGGT 模型推理；
#    2) 得到每帧的深度、外参（位姿）、内参等预测；
#    3) 使用重叠区域的点云进行相邻块的 Sim(3) 配准（估计尺度s、旋转R、平移t）；
#    4) （可选）做回环检测：DBoW2 或基于特征的 LoopDetector，得到回环约束；
#    5) 基于回环约束执行全局 Sim(3) 优化，减少长序列漂移；
#    6) 应用累计的 Sim(3) 变换，对齐所有块，导出点云和相机轨迹/内参等；
#    7) 管理中间文件，合并点云。
#  备注：
#    - 本注释版本仅添加注释，不更改任何代码逻辑或执行顺序；
#    - 新手可从 main 入口开始看流程，再回到各函数细节；
#    - 依赖的工具函数（如 weighted_align_point_maps 等）来自 loop_utils.sim3utils。
# ============================================================

import numpy as np
import argparse

import os
import glob
import threading
import torch
from tqdm.auto import tqdm  # 进度条库（在Notebook/终端自动适配）
import cv2

import gc  # 垃圾回收，用于清理内存/显存

# onnxruntime 用于可选的天空分割（本脚本中默认关闭 sky_mask）
try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

# 回环检测两种实现：LoopDetector（DNIO v2）或 DBoW2（RetrievalDBOW）
from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW

# VGGT 模型与其工具函数（图像预处理、位姿编码解码、几何工具）
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import numpy as np  # （重复导入，虽然冗余，但不影响功能）

# Sim(3) 回环优化器与一组 Sim(3)/点云相关工具函数
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *  # 包含 weighted_align_point_maps、compute_sim3_ab 等
from datetime import datetime

from PIL import Image  # 读图（配合 numpy/cv2）

# 无界面后端，便于服务器上保存图片
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from loop_utils.config_utils import load_config  # 读取 YAML 配置


def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
        去重函数：用于去除重复的回环段对（以 (chunk_idx_a, chunk_idx_b) 为键）
        只保留第一条出现的组合，且跳过 a==b 的无效回环。
    """
    seen = {}
    result = []

    for item in data_list:
        if item[0] == item[2]:
            # 自身块不构成回环，跳过
            continue

        key = (item[0], item[2])  # 以 (chunk_idx_a, chunk_idx_b) 作为唯一性判断

        if key not in seen.keys():
            seen[key] = True
            result.append(item)

    return result


class LongSeqResult:
    # 结果容器（本脚本中未直接使用到，预留给可能的合并结果结构）
    def __init__(self):
        self.combined_extrinsics = []  # 合并后的外参
        self.combined_intrinsics = []  # 合并后的内参
        self.combined_depth_maps = []  # 合并后的深度图
        self.combined_depth_confs = []  # 深度置信度
        self.combined_world_points = []  # 世界坐标点云
        self.combined_world_points_confs = []  # 点云置信度
        self.all_camera_poses = []  # 所有相机位姿（可能是列表形式）
        self.all_camera_intrinsics = []  # 所有相机内参


class VGGT_Long:
    # 核心类：完成加载、分块推理、对齐、回环、优化、导出等整套流程
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        # 关键超参数：分块大小与重叠帧数（影响显存、对齐稳定性）
        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.conf_threshold = 1.5  # 置信度阈值的初值（某些函数可能使用更细规则）
        self.seed = 42  # 随机种子（如需严格可复现，还需 cudnn 确定性设置）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 根据 GPU 架构选择 bfloat16 或 float16，用于 AMP 自动混合精度
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False  # 是否启用天空分割（默认 False）
        self.useDBoW = self.config['Model']['useDBoW']  # 回环使用 DBoW2 还是 DNIO v2

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        # 中间结果与输出目录
        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')  # 未对齐的每块预测
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')  # 应用 Sim(3) 后的对齐结果
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')  # 回环对的临时结果
        self.pcd_dir = os.path.join(save_dir, 'pcd')  # 导出的点云目录
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        # 用于最终导出相机位姿/内参（按块收集）
        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        # 是否在结束时删除临时文件，节省磁盘空间
        self.delete_temp_files = self.config['Model']['delete_temp_files']

        print('Loading model...')

        # 加载 VGGT 模型与权重（权重路径来自配置）
        self.model = VGGT()
        # 可选：从 URL 下载官方权重（此处注释掉）
        # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        _URL = self.config['Weights']['VGGT']
        state_dict = torch.load(_URL, map_location='cuda')
        self.model.load_state_dict(state_dict, strict=False)  # strict=False 允许部分键不匹配（兼容性更好）

        self.model.eval()  # 评估模式（禁用 Dropout 等）
        self.model = self.model.to(self.device)

        self.skyseg_session = None  # onnxruntime 的天空分割会话（默认未启用）

        # 如果启用天空分割，这里会加载 skyseg.onnx（当前注释）
        # if self.sky_mask:
        #     print('Loading skyseg.onnx...')
        #     # 若本地无文件则下载
        #     if not os.path.exists("skyseg.onnx"):
        #         print("Downloading skyseg.onnx...")
        #         download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")
        #     self.skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")

        self.chunk_indices = None  # 记录每个块对应的起止帧索引列表：[(begin_idx, end_idx), ...]

        self.loop_list = []  # 回环对列表（例如 [(1584, 139), ...] 表示帧1584与帧139构成回环）

        self.loop_optimizer = Sim3LoopOptimizer(self.config)  # 基于 Sim(3) 的图优化器

        self.sim3_list = []  # 邻接块之间的 Sim(3) 列表：[(s, R, t), ...]，顺序与块连接一致

        self.loop_sim3_list = []  # 回环约束对应的 Sim(3)：[(chunk_idx_a, chunk_idx_b, (s,R,t)), ...]

        self.loop_predict_list = []  # 存放回环对对应的模型预测结果（用于计算 a/b 相对Sim(3)）

        self.loop_enable = self.config['Model']['loop_enable']  # 是否启用回环

        # 初始化回环检索器（DBoW2 或 DNIO v2）
        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        print('init done.')

    def get_loop_pairs(self):
        # 根据配置选择回环检测方案，得到可能的回环帧对 self.loop_list
        if self.useDBoW:  # 使用 DBoW2 词袋模型做图像回环检索
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # 单通道灰度图转换为 RGB（三通道），以适配检索器
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori  # (H, W, 3)
                # 将图像缩小一半，加快检索速度
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                # 喂给检索器并登记该帧
                self.retrieval(frame, frame_id)
                # 基于阈值和重复次数确认是否触发回环候选
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'],
                                                   num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands  # 示例：cands = (812, 67) 表示帧812与帧67回环
                    self.retrieval.confirm_loop(i, j)  # 向内部结构确认并记录
                    self.retrieval.found.clear()  # 清空临时found，避免重复记录
                    self.loop_list.append(cands)  # 保存回环对

                # 将至多到 frame_id 的状态写盘（节省内存）
                self.retrieval.save_up_to(frame_id)

        else:  # 使用 LoopDetector（外部方法，标注“DNIO v2”）
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        # 处理一个“块”的图像：推理 -> 提取外参/内参/深度等 -> 保存结果到 .npy
        # range_1: 主块的 [start_idx, end_idx)
        # range_2: 若是回环拼接预测，这里会附加另一段范围的图像
        # is_loop: True 表示这是针对回环对的特殊拼接推理（用于估计 a/b 的相对Sim(3)）

        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            # 若进行回环合并预测，将两段图像拼接送入模型（便于估计两段的相对关系）
            chunk_image_paths += self.img_list[start_idx:end_idx]

        # 加载并预处理图像（尺寸、归一化、打包为 Tensor） -> 送往 GPU/CPU
        images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        print(f"Loaded {len(images)} images")

        # 模型输入检查：BCHW 且 3通道
        # images 形状: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        torch.cuda.empty_cache()
        with torch.no_grad():  # 推理不需要梯度，节省显存与加速
            # AMP 自动混合精度，依据前面 self.dtype 选择 bfloat16/float16
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(images)  # VGGT 前向，输出字典：包含 pose_enc/depth/... 等
        torch.cuda.empty_cache()

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        # 将模型的姿态编码（pose_enc）解码为 外参(Extrinsic, W2C/或C2W之一) 与 内参(Intrinsic K)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        print("Processing model outputs...")
        # 将张量转为 numpy 并 squeeze batch 维度（B=1时）
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        # 根据是否为回环预测，选择不同的落盘目录与文件名
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                # 普通块必须提供 chunk_idx，用于命名文件和记录位姿范围
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        # 对于普通块（非回环）且未拼接第二段时，记录该块的外参/内参到全局列表，便于最终导出
        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            intrinsics = predictions['intrinsic']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        # 深度可能是 [B, H, W]，此处再挤掉多余维度（与上面一致性）
        predictions['depth'] = np.squeeze(predictions['depth'])

        # 将整块预测字典保存为 .npy（包含 images/poses/depth/world_points 等）
        np.save(save_path, predictions)

        # 如果是回环预测或包含 range_2（拼接），返回预测结果（后续需要用于计算相对 Sim(3)）
        # 否则返回 None，避免占用内存
        return predictions if is_loop or range_2 is not None else None

    def process_long_sequence(self):
        # 整体流程：分块 -> （可选）回环块预测 -> 各块推理 -> 相邻块 Sim(3) -> 回环 Sim(3) -> 全局优化 -> 应用对齐 -> 导出
        if self.overlap >= self.chunk_size:
            # 参数合法性检查：重叠不能大于等于块大小
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        if len(self.img_list) <= self.chunk_size:
            # 图片数量不超过块大小时，只需一块
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            # 一般情况：步长 = 块大小 - 重叠
            step = self.chunk_size - self.overlap
            # 计算块的数量（整除上取）
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))

        # 如果开启回环：根据帧级回环对，生成“块级回环对”，并对这些对进行拼接预测，得到用于估计 a/b Sim(3) 的预测结果
        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            # 将帧级回环对映射到块级对，half_window 控制每个回环两端局部窗口长度
            loop_results = process_loop_list(self.chunk_indices,
                                             self.loop_list,
                                             half_window=int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)  # 去重
            print(loop_results)
            # 例如返回 (31, (1574, 1594), 2, (129, 149))：
            #   表示 chunk_idx_a=31 的区间 [1574,1594) 与 chunk_idx_b=2 的区间 [129,149)
            for item in loop_results:
                # 对这两个区间（拼接后）做一次模型推理，以便估计它们之间的相对Sim(3)
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)

                # 记录回环对与其预测（后续对齐）
                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap")

        # 遍历每一个块，逐块推理并落盘
        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        # 推理完成后可释放模型，节省显存
        del self.model  # Save GPU Memory
        torch.cuda.empty_cache()

        # 对齐所有相邻块：利用重叠区域的点云与置信度，加权估计 Sim(3)
        print("Aligning all the chunks...")
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Aligning {chunk_idx} and {chunk_idx + 1} (Total {len(self.chunk_indices) - 1})")
            # 读取相邻两块的未对齐预测
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"),
                                  allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"),
                                  allow_pickle=True).item()

            # 取两块的重叠帧范围的世界坐标点云与置信度
            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            # 以两段置信度中位数的 0.1 倍作为加权阈值，过滤低置信度点
            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            # 加权的点云对齐函数，求出 Sim(3)：尺度s、旋转R(3x3)、平移t(3,)
            s, R, t = weighted_align_point_maps(point_map1,
                                                conf1,
                                                point_map2,
                                                conf2,
                                                conf_threshold=conf_threshold,
                                                config=self.config)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)

            # 记录这一对相邻块之间的 Sim(3)
            self.sim3_list.append((s, R, t))

        # 若开启回环：对每个回环对，分别把回环拼接预测与原块局部做对齐，得到 a->loop 与 b->loop 的 Sim(3)，
        # 再组合成 a->b 的 Sim(3)，作为回环约束加入图优化
        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                print('chunk_a align')
                # 从回环拼接预测中，截取对应 chunk_a 的长度部分作为“loop参考”
                point_map_loop = item[1]['world_points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = item[1]['world_points_conf'][:chunk_a_range[1] - chunk_a_range[0]]
                # 计算 chunk_a 在自身块内的相对索引范围
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                print(self.chunk_indices[chunk_idx_a])
                print(chunk_a_range)
                print(chunk_a_rela_begin, chunk_a_rela_end)
                chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                                       allow_pickle=True).item()

                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]

                conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                s_a, R_a, t_a = weighted_align_point_maps(point_map_a,
                                                          conf_a,
                                                          point_map_loop,
                                                          conf_loop,
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                print('chunk_a align')
                # 同理，对 chunk_b 做对齐（截取回环拼接预测中对应 b 的尾部段）
                point_map_loop = item[1]['world_points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['world_points_conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                print(self.chunk_indices[chunk_idx_b])
                print(chunk_b_range)
                print(chunk_b_rela_begin, chunk_b_rela_end)
                chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                                       allow_pickle=True).item()

                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]

                conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                s_b, R_b, t_b = weighted_align_point_maps(point_map_b,
                                                          conf_b,
                                                          point_map_loop,
                                                          conf_loop,
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                print('a -> b SIM 3')
                # 将 a->loop 与 b->loop 组合得到 a->b 的 Sim(3)（利用 sim3 群的组合规则）
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)

                # 记录回环约束：(chunk_idx_a, chunk_idx_b, Sim3_ab)
                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        # 若开启回环：将相邻块的 sim3 边 + 回环边 一起丢给优化器，做全局 Sim(3) 图优化
        if self.loop_enable:
            # 优化前后，提取绝对位姿（将相对 sim3 链式展开到全局坐标，用于可视化）
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            # 提取平面(x,z)用于绘图（假定 y 轴为上）
            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]

            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            # 保存优化前后轨迹与回环边连线的可视化图
            plt.figure(figsize=(8, 6))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25, label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35, label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.output_dir, 'sim3_opt_result.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print('Apply alignment')
        # 将相邻 sim3 链式累计到第一个块的坐标系（得到每个块相对第0块的累计 Sim(3)）
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f'Applying {chunk_idx + 1} -> {chunk_idx} (Total {len(self.chunk_indices) - 1})')
            s, R, t = self.sim3_list[chunk_idx]

            # 读取第 chunk_idx+1 个块的未对齐结果
            chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"),
                                 allow_pickle=True).item()

            # 对该块的世界点云应用累计的 Sim(3)，把它对齐到第0块坐标系
            chunk_data['world_points'] = apply_sim3_direct(chunk_data['world_points'], s, R, t)

            # 保存对齐后的该块结果
            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx + 1}.npy")
            np.save(aligned_path, chunk_data)

            if chunk_idx == 0:
                # 第0块本身就是参考系，直接把未对齐的0号块拷贝到“已对齐”目录（数值未变）
                chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"),
                                           allow_pickle=True).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)

            # 从已对齐的块中读取数据，准备导出点云（PLY）
            aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy"),
                                         allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first

            # 组装点云：points(H*W,3)、colors(H*W,3)、confs(H*W,)
            points = aligned_chunk_data['world_points'].reshape(-1, 3)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = aligned_chunk_data['world_points_conf'].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx}_pcd.ply')
            # 根据平均置信度 * 系数 做阈值筛选 + 下采样，存盘
            save_confident_pointcloud_batch(
                points=points,  # shape: (N, 3)
                colors=colors,  # shape: (N, 3)
                confs=confs,  # shape: (N,)
                output_path=ply_path,
                conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )

        # 在对齐完成后，根据记录的外参/内参与累计 sim3，导出全局相机轨迹与内参
        self.save_camera_poses()

        print('Done.')

    def run(self):
        # 主流程：加载图像 -> 回环对（可选）-> 分块处理
        print(f"Loading images from {self.img_dir}...")
        # 读取目录下所有 jpg/png，并排序（按文件名升序）
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                               glob.glob(os.path.join(self.img_dir, "*.png")))
        # print(self.img_list)
        if len(self.img_list) == 0:
            # 若目录为空，直接报错提醒
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        # 若开启回环，先预计算回环帧对
        if self.loop_enable:
            self.get_loop_pairs()

            # 用完检索器/检测器后，释放资源
            if self.useDBoW:
                self.retrieval.close()  # 释放CPU内存
                gc.collect()
            else:
                del self.loop_detector  # 释放GPU内存
        torch.cuda.empty_cache()

        # 进入长序列分块处理主流程
        self.process_long_sequence()

    def save_camera_poses(self):
        '''
        导出所有相机位姿与内参，并保存一个简单的点云文件来可视化相机中心位置。
        - camera_poses.txt：每行是一个 4x4 的 C2W 矩阵（按行展平为16个数）
        - intrinsic.txt：每行四个值 fx fy cx cy（来自3x3内参矩阵）
        - camera_poses.ply：将相机中心位置以点的形式写入PLY（颜色固定为 chunk_colors[0]）
        '''
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        # 预分配列表，长度与帧数一致
        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        # 先处理第0块：其坐标系作为全局参考
        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i]  # 这里 extrinsic 视为 W2C（世界到相机）
            c2w = np.linalg.inv(w2c)  # 取逆得到 C2W（相机到世界）
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]

        # 其余块：先由记录的 sim3_list（累计到第0块坐标）得到变换 S，然后对每帧的 C2W 左乘 S
        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[chunk_idx - 1]  # 此时 sim3_list 已经用 accumulate 对齐到第0块

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # 注意左乘顺序：先将该块坐标变换到全局（第0块参考系）
                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[i]

        # 写出 C2W 到 txt（每行16个数：4x4 展平）
        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')

        print(f"Camera poses saved to {poses_path}")

        # 写出内参到 txt（每行 fx fy cx cy）
        intrinsics_path = os.path.join(self.output_dir, 'intrinsic.txt')
        with open(intrinsics_path, 'w') as f:
            for intrinsic in all_intrinsics:
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                f.write(f'{fx} {fy} {cx} {cy}\n')

        print(f"Camera intrinsics saved to {intrinsics_path}")

        # 写一个简单的 PLY，把每个相机的中心位置（C2W 的平移）写成点云
        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            # PLY 头
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            color = chunk_colors[0]  # 这里固定用了第一个颜色（未按块区分颜色）
            for pose in all_poses:
                position = pose[:3, 3]  # C2W 的平移即相机中心在世界坐标的位置
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        '''
            清理中间临时文件，并统计回收的磁盘空间。
            将删除以下目录中的文件：
            - 未对齐结果目录 _tmp_results_unaligned
            - 已对齐结果目录 _tmp_results_aligned
            - 回环临时结果目录 _tmp_results_loop

            大致空间占用参考：
            ~50 GiB 对应 KITTI 00（4500 帧）,
            ~35 GiB 对应 KITTI 05（2700 帧）,
            ~5 GiB 对应 300 帧短序列。
        '''
        if not self.delete_temp_files:
            return

        total_space = 0

        print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print('Deleting temp files done.')

        print(f"Saved disk space: {total_space / 1024 / 1024 / 1024:.4f} GiB")


import shutil


def copy_file(src_path, dst_dir):
    # 将配置文件复制到实验输出目录，便于复现实验记录
    try:
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path

    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")


if __name__ == '__main__':

    # 命令行参数解析：图像目录、配置文件路径
    parser = argparse.ArgumentParser(description='VGGT-Long')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/base_config.yaml',
                        help='Image path')
    args = parser.parse_args()

    # 读取 YAML 配置（包含模型超参、路径、回环参数等）
    config = load_config(args.config)

    image_dir = args.image_dir
    path = image_dir.split("/")
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'

    # 以 “exps/<image_dir替换斜杠为下划线>/<时间戳>/” 作为本次实验输出目录
    save_dir = os.path.join(
        exp_dir, image_dir.replace("/", "_"), current_datetime
    )

    # 另一种命名方式（按路径末级三段组合）——当前注释掉
    # save_dir = os.path.join(
    #     exp_dir, path[-3] + "_" + path[-2] + "_" + path[-1], current_datetime
    # )

    # 若目录不存在则创建，并把当前使用的配置文件复制过去做留档
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')
        copy_file(args.config, save_dir)

    # 若选择 numba 对齐方法，这里会先做一次“预热”，便于加速（函数来自 sim3utils）
    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    # 实例化流程管理器并运行
    vggt_long = VGGT_Long(image_dir, save_dir, config)
    vggt_long.run()
    vggt_long.close()

    # 释放对象与显存
    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()

    # 合并各块导出的点云（ply）为一个总的 combined_pcd.ply
    all_ply_path = os.path.join(save_dir, f'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print('VGGT Long done.')
    sys.exit()
