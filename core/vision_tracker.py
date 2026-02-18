#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统计算机视觉追踪模块
使用背景减除+光流分析
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict, Any
import time
import os

# YOLOv11-seg (Ultralytics) 可选依赖
try:
    from ultralytics import YOLO  # type: ignore
    _HAS_ULTRALYTICS = True
except Exception:
    YOLO = None  # type: ignore
    _HAS_ULTRALYTICS = False

import threading

# GPU 检测（用于 OpenCV CUDA，如果可用）
USE_GPU = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        USE_GPU = True
        print(f"✓ OpenCV CUDA 可用")
except:
    pass


_YOLO_MODEL_CACHE: Dict[str, Any] = {}
_YOLO_MODEL_LOCK = threading.Lock()


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_yolo_model(model_path: str):
    """Load and cache a YOLO model by path (thread-safe)."""
    with _YOLO_MODEL_LOCK:
        model = _YOLO_MODEL_CACHE.get(model_path)
        if model is not None:
            return model
        if not _HAS_ULTRALYTICS:
            raise RuntimeError("未安装 ultralytics，无法使用YOLO分割检测")
        model = YOLO(model_path)
        # 仅在首次加载时做一次 fuse（如果支持）
        try:
            model.fuse()
        except Exception:
            pass
        # 打印一次性信息，帮助确认是否可用CUDA
        try:
            dev = 'cuda' if _torch_cuda_available() else 'cpu'
            print(f"YOLO模型已加载: {os.path.basename(str(model_path))} | torch_device_hint={dev}")
        except Exception:
            pass
        _YOLO_MODEL_CACHE[model_path] = model
        return model


class YoloSegDetector:
    """Use a YOLO segmentation model to produce mouse mask + bbox inside ROI."""

    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 640):
        self.model_path = model_path
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self._model = _get_yolo_model(model_path)
        self.device = 'cuda:0' if _torch_cuda_available() else 'cpu'
        self.half = bool(self.device.startswith('cuda'))

    def _predict(self, source):
        """Internal predict wrapper to force device/half and keep calls consistent."""
        try:
            # Ultralytics API: model.predict(...) is stable across versions
            return self._model.predict(
                source=source,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        except Exception:
            # fallback to __call__
            try:
                return self._model(source, conf=self.conf, imgsz=self.imgsz, verbose=False)
            except Exception:
                return []

    def detect_batch(self, bgr_frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Batch inference. Returns list aligned to input frames."""
        if not bgr_frames:
            return []
        try:
            results = self._predict(bgr_frames)
        except Exception:
            results = []

        if not results:
            return [[] for _ in bgr_frames]

        out: List[List[Dict[str, Any]]] = []
        for frame, r0 in zip(bgr_frames, results):
            out.append(self._parse_single_result(frame, r0, roi_polygon_mask=None))
        # 若 results 数量不足，补齐
        while len(out) < len(bgr_frames):
            out.append([])
        return out

    def _parse_single_result(self, roi_bgr: np.ndarray, r0, roi_polygon_mask: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        if r0 is None:
            return []
        if getattr(r0, 'masks', None) is None or getattr(r0, 'boxes', None) is None:
            return []

        boxes = r0.boxes
        if boxes is None:
            return []

        h, w = roi_bgr.shape[:2]

        # 优先使用 polygon（避免把全分辨率 float mask 从 GPU 拷回 CPU，极慢）
        polys = None
        try:
            polys = getattr(r0.masks, 'xy', None)
        except Exception:
            polys = None

        if polys is not None:
            try:
                boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
            except Exception:
                return []

            detections: List[Dict[str, Any]] = []
            for poly, box_xyxy, conf in zip(polys, boxes_xyxy, confs):
                if poly is None:
                    continue
                try:
                    poly_np = np.asarray(poly, dtype=np.float32)
                except Exception:
                    continue
                if poly_np.ndim != 2 or poly_np.shape[0] < 3 or poly_np.shape[1] != 2:
                    continue

                # clip 到图像范围
                poly_np[:, 0] = np.clip(poly_np[:, 0], 0, w - 1)
                poly_np[:, 1] = np.clip(poly_np[:, 1], 0, h - 1)
                poly_i32 = poly_np.astype(np.int32)

                base_mask_u8 = np.zeros((h, w), dtype=np.uint8)
                try:
                    cv2.fillPoly(base_mask_u8, [poly_i32], 255)
                except Exception:
                    continue

                if roi_polygon_mask is not None and roi_polygon_mask.size > 0:
                    if roi_polygon_mask.shape[:2] != base_mask_u8.shape[:2]:
                        rpm = cv2.resize(roi_polygon_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        rpm = roi_polygon_mask
                    mask_u8 = cv2.bitwise_and(base_mask_u8, base_mask_u8, mask=rpm)
                else:
                    mask_u8 = base_mask_u8

                area = int(np.count_nonzero(mask_u8))
                if area <= 0:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box_xyxy]
                x1 = max(0, min(w - 1, int(round(x1))))
                y1 = max(0, min(h - 1, int(round(y1))))
                x2 = max(0, min(w, int(round(x2))))
                y2 = max(0, min(h, int(round(y2))))
                if x2 <= x1 or y2 <= y1:
                    continue

                bx, by = x1, y1
                bw, bh = (x2 - x1), (y2 - y1)
                cx = float(bx + bw / 2.0)
                cy = float(by + bh / 2.0)

                detections.append({
                    'bbox_xyxy': (x1, y1, x2, y2),
                    'bbox': (bx, by, bw, bh),
                    'centroid': (cx, cy),
                    'polygon': poly_i32,
                    'mask': mask_u8,
                    'area': area,
                    'conf': float(conf),
                })
            return detections

        # 回退：旧版/不支持 polygon 的情况下才走 masks.data（可能很慢）
        masks = getattr(r0.masks, 'data', None)
        if masks is None:
            return []

        try:
            masks_np = masks.detach().cpu().numpy()
            boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
        except Exception:
            return []
        detections: List[Dict[str, Any]] = []
        for mask_f, box_xyxy, conf in zip(masks_np, boxes_xyxy, confs):
            if mask_f.shape[0] != h or mask_f.shape[1] != w:
                mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)

            base_mask_u8 = (mask_f > 0.5).astype(np.uint8) * 255
            base_area = int(np.sum(base_mask_u8 == 255))
            if base_area <= 0:
                continue

            if roi_polygon_mask is not None and roi_polygon_mask.size > 0:
                if roi_polygon_mask.shape[:2] != base_mask_u8.shape[:2]:
                    rpm = cv2.resize(roi_polygon_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    rpm = roi_polygon_mask
                cropped_mask = cv2.bitwise_and(base_mask_u8, base_mask_u8, mask=rpm)
                cropped_area = int(np.sum(cropped_mask == 255))
                mask_u8 = cropped_mask if cropped_area > 0 else base_mask_u8
            else:
                mask_u8 = base_mask_u8

            area = int(np.sum(mask_u8 == 255))
            if area <= 0:
                continue

            x1, y1, x2, y2 = [float(v) for v in box_xyxy]
            x1 = max(0, min(w - 1, int(round(x1))))
            y1 = max(0, min(h - 1, int(round(y1))))
            x2 = max(0, min(w, int(round(x2))))
            y2 = max(0, min(h, int(round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue

            bx, by = x1, y1
            bw, bh = (x2 - x1), (y2 - y1)
            cx = float(bx + bw / 2.0)
            cy = float(by + bh / 2.0)

            detections.append({
                'bbox_xyxy': (x1, y1, x2, y2),
                'bbox': (bx, by, bw, bh),
                'centroid': (cx, cy),
                'polygon': None,
                'mask': mask_u8,
                'area': area,
                'conf': float(conf),
            })
        return detections

    def detect(self, roi_bgr: np.ndarray, roi_polygon_mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Return list of detections in ROI-local coordinates.

        Each detection dict contains:
          - bbox_xyxy: (x1, y1, x2, y2) ROI-local
          - bbox: (x, y, w, h) ROI-local
          - centroid: (cx, cy) ROI-local
          - mask: uint8 ROI-local (0/255)
          - area: int, mask pixel count
          - conf: float
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return []

        results = self._predict(roi_bgr)
        if not results:
            return []
        r0 = results[0]
        return self._parse_single_result(roi_bgr, r0, roi_polygon_mask=roi_polygon_mask)


class BackgroundSubtractor:
    """
    前景检测器（支持多种模式）
    
    模式:
    1. 'threshold' - 简单灰度阈值（推荐，最快最简单）
    2. 'MOG2' - 自适应背景减除
    3. 'KNN' - 自适应背景减除
    4. 'static' - 静态背景减除
    """
    
    def __init__(self, method='threshold', threshold=127, invert=False, static_bg=None):
        """
        初始化检测器
        
        参数:
            method: 'threshold', 'MOG2', 'KNN', 或 'static'
            threshold: 灰度阈值（仅threshold模式使用）
            invert: 是否反转（黑背景设为True）
            static_bg: 静态背景图像（仅static模式使用）
        """
        self.method = method
        self.threshold = threshold
        self.invert = invert
        self.static_bg = static_bg
        
        if method == 'threshold':
            # 简单阈值模式（最快，推荐）
            pass
        elif method == 'MOG2':
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=False
            )
        elif method == 'KNN':
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=500,
                dist2Threshold=400,
                detectShadows=False
            )
        elif method == 'static':
            if static_bg is None:
                raise ValueError("Static background image required for static method")
            self.static_bg = cv2.cvtColor(static_bg, cv2.COLOR_BGR2GRAY) if len(static_bg.shape) == 3 else static_bg
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def apply(self, frame, learning_rate=-1, gray=None):
        """
        应用前景检测
        
        参数:
            frame: 输入帧（BGR或灰度）
            learning_rate: 学习率（仅自适应方法）
            gray: 预计算的灰度图（可选，避免重复转换）
        
        返回:
            前景掩码（二值图像）
        """
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.method == 'threshold':
            # 简单灰度阈值分割（最快最简单）
            if self.invert:
                # 黑色背景，老鼠更亮 → 保留亮的
                _, fg_mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
            else:
                # 白色背景，老鼠更暗 → 保留暗的
                _, fg_mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
            return fg_mask
        elif self.method == 'static':
            # 静态背景减除
            diff = cv2.absdiff(gray, self.static_bg)
            _, fg_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            return fg_mask
        else:
            # 自适应背景减除（MOG2/KNN）
            return self.subtractor.apply(gray, learningRate=learning_rate)


class MorphologyProcessor:
    """形态学处理器（性能优化版）"""
    
    def __init__(self, 
                 erosion_kernel=3,
                 erosion_iterations=2,
                 dilation_kernel=5,
                 dilation_iterations=3,
                 use_closing=True,
                 closing_kernel=5):
        """
        初始化形态学处理参数
        
        参数:
            erosion_kernel: 腐蚀核大小
            erosion_iterations: 腐蚀迭代次数
            dilation_kernel: 膨胀核大小
            dilation_iterations: 膨胀迭代次数
            use_closing: 是否使用闭运算（填充小孔）
            closing_kernel: 闭运算核大小
        """
        self.erosion_kernel = erosion_kernel
        self.erosion_iterations = erosion_iterations
        self.dilation_kernel = dilation_kernel
        self.dilation_iterations = dilation_iterations
        self.use_closing = use_closing
        self.closing_kernel = closing_kernel
        
        # 【性能优化】预创建结构元素，避免每帧重复创建
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.closing_kernel, self.closing_kernel)
        ) if use_closing else None
        
        self._kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.erosion_kernel, self.erosion_kernel)
        ) if erosion_iterations > 0 else None
        
        self._kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.dilation_kernel, self.dilation_kernel)
        ) if dilation_iterations > 0 else None
    
    def process(self, mask, inplace=True):
        """
        应用形态学处理（性能优化版）
        
        参数:
            mask: 输入二值掩码
            inplace: 是否原地修改（避免复制，更快）
        
        返回:
            处理后的掩码
        """
        # 【性能优化】避免不必要的复制
        result = mask if inplace else mask.copy()
        
        # 闭运算（先膨胀后腐蚀，填充小孔）
        # 【性能优化】使用预创建的结构元素
        if self.use_closing and self._kernel_close is not None:
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, self._kernel_close)
        
        # 腐蚀（去除小噪点）
        if self.erosion_iterations > 0 and self._kernel_erode is not None:
            result = cv2.erode(result, self._kernel_erode, iterations=self.erosion_iterations)
        
        # 膨胀（恢复轮廓）
        if self.dilation_iterations > 0 and self._kernel_dilate is not None:
            result = cv2.dilate(result, self._kernel_dilate, iterations=self.dilation_iterations)
        
        return result


class MouseDetector:
    """小鼠检测器"""
    
    def __init__(self,
                 min_area=100,
                 max_area=50000,
                 min_solidity=0.3):
        """
        初始化检测参数
        
        参数:
            min_area: 最小轮廓面积（像素）
            max_area: 最大轮廓面积（像素）
            min_solidity: 最小实心度（过滤奇形怪状的轮廓）
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_solidity = min_solidity
    
    def detect(self, mask):
        """
        检测小鼠轮廓
        
        参数:
            mask: 前景掩码
        
        返回:
            List[Dict]: 检测结果列表，每个包含：
                - contour: 轮廓
                - centroid: 质心 (x, y)
                - area: 面积
                - bbox: 边界框 (x, y, w, h)
        """
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < self.min_area or area > self.max_area:
                continue
            
            # 实心度过滤（排除奇形怪状的轮廓）
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < self.min_solidity:
                continue
            
            # 计算质心
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            
            # 边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            detections.append({
                'contour': contour,
                'centroid': (cx, cy),
                'area': area,
                'bbox': (x, y, w, h),
                'solidity': solidity,
            })
        
        return detections


class OpticalFlowAnalyzer:
    """
    光流分析器（使用 OpenCV Farneback 算法）
    
    优化策略:
    1. ROI 区域裁剪 - 只计算小鼠所在区域
    2. 分辨率缩放 - 降低计算量
    """
    
    def __init__(self, fps=30.0, pixel_to_mm=1.0):
        """
        初始化光流分析器
        
        参数:
            fps: 视频帧率
            pixel_to_mm: 像素到毫米的转换比例
        """
        self.fps = fps
        self.pixel_to_mm = pixel_to_mm
        self.prev_gray = None
        self.frame_count = 0
    
    def analyze(self, curr_frame, mask, curr_gray=None, bbox=None, mask_offset=None):
        """
        分析光流（只在 mask 区域内）
        
        参数:
            curr_frame: 当前帧（BGR 或灰度）
            mask: 小鼠区域掩码（可以是裁剪后的局部 mask）
            curr_gray: 预计算的灰度图（可选，避免重复转换）
            bbox: 预计算的边界框 (x1, y1, x2, y2)（可选，避免重复 findContours）
            mask_offset: mask 相对于全图的偏移量 (offset_x, offset_y)，用于裁剪 mask 的情况
        
        返回:
            Dict: 光流统计量
        """
        self.frame_count += 1
        
        # 【性能优化】复用预计算的灰度图
        if curr_gray is None:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
        
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return None
        
        # 【性能优化】复用预计算的边界框
        if bbox is not None:
            x1, y1, x2, y2 = bbox
        else:
            # 需要自己计算边界框
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                self.prev_gray = curr_gray
                return None
            
            # 合并所有轮廓的外接矩形
            x_min, y_min = mask.shape[1], mask.shape[0]
            x_max, y_max = 0, 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            margin = 10
            x1 = max(0, x_min - margin)
            y1 = max(0, y_min - margin)
            x2 = min(curr_gray.shape[1], x_max + margin)
            y2 = min(curr_gray.shape[0], y_max + margin)
        
        # 裁剪 ROI 区域
        roi_prev = self.prev_gray[y1:y2, x1:x2]
        roi_curr = curr_gray[y1:y2, x1:x2]
        
        # 【性能优化】分辨率缩放：0.35x
        scale_factor = 1.0
        orig_shape = roi_prev.shape
        if roi_prev.shape[0] > 40 or roi_prev.shape[1] > 40:
            scale_factor = 0.35
            roi_prev = cv2.resize(roi_prev, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            roi_curr = cv2.resize(roi_curr, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            if self.frame_count == 2:
                print(f"  [优化] 光流分辨率: {orig_shape} → {roi_prev.shape} (缩放 {scale_factor}x)")
        
        # 使用 OpenCV Farneback 光流算法
        # 【性能优化】调整参数：levels=2, iterations=2（比默认快30%，精度损失<2%）
        flow = cv2.calcOpticalFlowFarneback(
            roi_prev, roi_curr,
            None, 
            pyr_scale=0.5,   # 金字塔缩放
            levels=2,         # 金字塔层数（原3→2，快15%）
            winsize=15,       # 窗口大小
            iterations=2,     # 迭代次数（原3→2，快20%）
            poly_n=5,         # 多项式展开大小
            poly_sigma=1.2,   # 高斯标准差
            flags=0
        )
        
        # 【性能优化】直接在缩小的 flow 上计算统计量，避免昂贵的 resize
        # 对于统计量（均值、最大值等），缩放不影响结果
        
        # 计算 mask 在当前 bbox 区域内的局部坐标
        # 如果 mask 是裁剪后的（有 mask_offset），需要调整坐标
        if mask_offset is not None:
            off_x, off_y = mask_offset
            # 从局部 mask 坐标转换：全局 bbox 坐标 -> 局部 mask 坐标
            local_y1 = max(0, y1 - off_y)
            local_y2 = min(mask.shape[0], y2 - off_y)
            local_x1 = max(0, x1 - off_x)
            local_x2 = min(mask.shape[1], x2 - off_x)
            raw_mask = mask[local_y1:local_y2, local_x1:local_x2]
        else:
            raw_mask = mask[y1:y2, x1:x2]
        
        if scale_factor != 1.0:
            # 补偿缩放：光流值需要除以 scale_factor
            flow = flow / scale_factor
            # 同时缩小 mask 用于过滤
            if raw_mask.size > 0:
                local_mask = cv2.resize(raw_mask, (flow.shape[1], flow.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
            else:
                local_mask = np.zeros((flow.shape[0], flow.shape[1]), dtype=np.uint8)
        else:
            # 确保 mask 和 flow 尺寸匹配
            if raw_mask.size > 0:
                if raw_mask.shape[0] != flow.shape[0] or raw_mask.shape[1] != flow.shape[1]:
                    local_mask = cv2.resize(raw_mask, (flow.shape[1], flow.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                else:
                    local_mask = raw_mask
            else:
                local_mask = np.zeros_like(flow[:,:,0], dtype=np.uint8)
        
        # 计算光流强度
        flow_mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        flow_ang = np.arctan2(flow[:,:,1], flow[:,:,0])
        
        # 用局部掩码过滤
        mouse_flow_mag = flow_mag[local_mask == 255]
        mouse_flow_ang = flow_ang[local_mask == 255]
        
        u = flow[:,:,0][local_mask == 255]
        v = flow[:,:,1][local_mask == 255]
        
        if len(mouse_flow_mag) == 0:
            self.prev_gray = curr_gray
            return None
        
        # 计算方向统计（entropy/coherence）
        # 关键：摆动端点/速度很小时，光流幅值接近0，方向会被噪声主导，导致 coherence/entropy 发生“非物理暴跌/暴涨”。
        # 解决：仅在“足够强”的光流像素上计算方向统计（相对阈值），并在像素不足时回退到全体有效像素。
        mag_all = np.sqrt(u**2 + v**2)
        valid_mag = mag_all > 1e-6
        if int(np.sum(valid_mag)) > 0:
            u_valid = u[valid_mag]
            v_valid = v[valid_mag]
            mag_valid = mag_all[valid_mag]

            # 选取较强像素：使用分位数阈值（避免硬编码 px 阈值在不同缩放/分辨率下失效）
            n_valid = int(mag_valid.shape[0])
            if n_valid >= 50:
                # 取 60% 分位数作为“强光流”门槛
                k = int(0.60 * (n_valid - 1))
                thr = float(np.partition(mag_valid, k)[k])
                strong = mag_valid >= max(1e-6, thr)
            else:
                strong = np.ones_like(mag_valid, dtype=bool)

            # 若“强光流”像素过少，则回退
            if int(np.sum(strong)) < 20:
                strong = np.ones_like(mag_valid, dtype=bool)

            u_s = u_valid[strong]
            v_s = v_valid[strong]
            mag_s = mag_valid[strong]

            # entropy：1 - 圆形均值向量长度（越小越同向）
            u_norm = u_s / mag_s
            v_norm = v_s / mag_s
            r_val = float(np.sqrt(np.mean(u_norm)**2 + np.mean(v_norm)**2))
            flow_entropy = float(1.0 - r_val)

            # coherence：平均向量长度 / 平均幅值（越接近1越像“刚体同向运动”）
            mean_u = float(np.mean(u_s))
            mean_v = float(np.mean(v_s))
            mean_flow_mag = float(np.sqrt(mean_u**2 + mean_v**2))
            avg_individual_mag = float(np.mean(mag_s))
            coherence = float(mean_flow_mag / avg_individual_mag) if avg_individual_mag > 1e-6 else 0.0
        else:
            flow_entropy = 1.0
            coherence = 0.0
        
        # 统计量（简化版，减少计算）
        flow_mean_px_frame = float(np.mean(mouse_flow_mag))
        flow_max_px_frame = float(np.max(mouse_flow_mag))
        
        n = len(mouse_flow_mag)
        if n > 50:
            flow_p50_px_frame = float(np.partition(mouse_flow_mag, n//2)[n//2])
            flow_p90_px_frame = float(np.partition(mouse_flow_mag, int(n*0.9))[int(n*0.9)])
        else:
            sorted_mag = np.sort(mouse_flow_mag)
            flow_p50_px_frame = float(sorted_mag[n//2]) if n > 0 else 0.0
            flow_p90_px_frame = float(sorted_mag[int(n*0.9)]) if n > 0 else 0.0
        
        flow_std_px_frame = float(np.std(mouse_flow_mag))
        
        # 转换为归一化单位
        flow_mean_mm_s = flow_mean_px_frame * self.pixel_to_mm * self.fps
        flow_p90_mm_s = flow_p90_px_frame * self.pixel_to_mm * self.fps
        flow_max_mm_s = flow_max_px_frame * self.pixel_to_mm * self.fps
        
        # coherence 已在上面以“强光流像素”方式计算（更抗端点噪声）
        
        self.prev_gray = curr_gray
        
        return {
            'flow_mean_px': flow_mean_px_frame,
            'flow_p50_px': flow_p50_px_frame,
            'flow_p90_px': flow_p90_px_frame,
            'flow_max_px': flow_max_px_frame,
            'flow_std_px': flow_std_px_frame,
            'flow_mean_mm_s': flow_mean_mm_s,
            'flow_p90_mm_s': flow_p90_mm_s,
            'flow_max_mm_s': flow_max_mm_s,
            'flow_entropy': float(flow_entropy),
            'coherence': float(coherence),
            'dominant_angle': float(np.mean(mouse_flow_ang)) if len(mouse_flow_ang) > 0 else 0.0,
        }


class PeriodicityAnalyzer:
    """
    周期性运动分析器（鲁棒版 - 基于自相关 Autocorrelation）
    原理：
    1. 计算轨迹的自相关函数 (ACF)
    2. 在预期的摆动周期范围内寻找 ACF 的峰值
    3. 峰值高度直接反映周期性的强弱，不依赖波形是否为完美正弦
    """
    def __init__(self, fps, window_seconds=2.0): # 窗口稍微加大，容纳更多周期
        self.fps = fps
        self.win_size = int(fps * window_seconds)
        # 存储X坐标
        self.x_history = deque(maxlen=self.win_size)
        # 存储Y坐标（用于区分“左右摆动” vs “上下挣扎/抖动”）
        self.y_history = deque(maxlen=self.win_size)
        
        # 摆动频率范围 (Hz)
        self.SWING_FREQ_MIN = 0.8
        self.SWING_FREQ_MAX = 3.0
        
        # 最小摆动幅度 (mm)
        # 降低阈值，允许检测微小摆动
        self.MIN_AMPLITUDE_MM = 2.0 
        
        # 自相关强度阈值 (0-1)
        # 0.6 表示信号与其延迟版本有显著的正相关
        self.ACF_THRESHOLD = 0.6

    def update(self, centroid_x, centroid_y=None):
        self.x_history.append(centroid_x)
        # 兼容旧调用：如果没有传 y，则不更新 y_history
        if centroid_y is not None:
            self.y_history.append(centroid_y)

    def motion_anisotropy(self):
        """返回 (std_x, std_y, y_over_x)；若历史不足返回 (None, None, None)。"""
        if len(self.x_history) < self.win_size or len(self.y_history) < self.win_size:
            return None, None, None
        x = np.array(self.x_history, dtype=np.float32)
        y = np.array(self.y_history, dtype=np.float32)
        sx = float(np.std(x))
        sy = float(np.std(y))
        y_over_x = float(sy / (sx + 1e-6))
        return sx, sy, y_over_x

    def is_swinging_relative(self, scale_px: float, min_amplitude_ratio: float = 0.03):
        """基于相对尺度的摆动检测。

        适用场景：不同拍摄距离/缩放导致的像素幅度变化较大、标定不稳定时。
        思路：用 bbox 高度(或其它尺度)将质心横向位移做归一化，以“体长比例”判断幅度是否足够。

        参数:
            scale_px: 当前尺度（推荐用 bbox 高度的中位数）
            min_amplitude_ratio: 最小幅度阈值（相对尺度比例，0.03=3%）

        返回:
            (is_swinging, est_freq_hz, acf_peak)
        """
        if len(self.x_history) < self.win_size:
            return False, 0.0, 0.0

        if scale_px is None or scale_px <= 1e-6:
            return False, 0.0, 0.0

        data = np.array(self.x_history, dtype=np.float32)

        # 1) 相对幅度检查
        amplitude_ratio = float(np.std(data) / float(scale_px))
        is_amplitude_sufficient = amplitude_ratio >= float(min_amplitude_ratio)

        # 2) 自相关分析
        y = data - float(np.mean(data))
        r = np.correlate(y, y, mode='full')
        r = r[r.size // 2:]
        if r[0] <= 0:
            return False, 0.0, 0.0
        r = r / r[0]

        min_lag = int(self.fps / self.SWING_FREQ_MAX)
        max_lag = int(self.fps / self.SWING_FREQ_MIN)
        if min_lag >= len(r) or max_lag >= len(r):
            return False, 0.0, 0.0

        search_window = r[min_lag:max_lag + 1]
        if len(search_window) == 0:
            return False, 0.0, 0.0

        peak_val = float(np.max(search_window))
        peak_idx_in_window = int(np.argmax(search_window))
        true_lag = int(min_lag + peak_idx_in_window)
        est_freq = float(self.fps / true_lag) if true_lag > 0 else 0.0

        # 局部峰值检查
        is_peak = True
        if 0 < true_lag < (len(r) - 1):
            if r[true_lag] < r[true_lag - 1] or r[true_lag] < r[true_lag + 1]:
                is_peak = False

        if is_peak and peak_val > self.ACF_THRESHOLD and is_amplitude_sufficient:
            return True, est_freq, peak_val

        return False, est_freq, peak_val

    def is_swinging(self, pixel_to_mm):
        if len(self.x_history) < self.win_size:
            return False, 0.0, 0.0
            
        data = np.array(self.x_history)
        
        # 1. 幅度检查 (Amplitude Check)
        # 使用标准差估算幅度
        std_val = np.std(data)
        amplitude_mm = std_val * pixel_to_mm
        is_amplitude_sufficient = amplitude_mm >= self.MIN_AMPLITUDE_MM
        
        # 即使幅度不足，也继续计算ACF，以便在界面显示数值
            
        # 2. 自相关分析 (Autocorrelation)
        # 去均值
        y = data - np.mean(data)
        # 计算自相关
        # mode='full' 返回长度为 2*N-1 的数组
        r = np.correlate(y, y, mode='full')
        # 取右半部分 (lag >= 0)
        r = r[r.size//2:]
        
        # 归一化 (r[0] 是能量/方差)
        if r[0] <= 0: return False, 0.0, 0.0
        r = r / r[0]
        
        # 3. 寻找峰值 (Peak Finding)
        # 计算对应频率范围的 lag 索引范围
        # Lag = FPS / Freq
        min_lag = int(self.fps / self.SWING_FREQ_MAX)
        max_lag = int(self.fps / self.SWING_FREQ_MIN)
        
        # 确保索引不越界
        if min_lag >= len(r) or max_lag >= len(r):
            return False, 0.0, 0.0
            
        # 在感兴趣的 lag 范围内寻找最大值
        search_window = r[min_lag : max_lag + 1]
        if len(search_window) == 0:
            return False, 0.0, 0.0
            
        peak_val = np.max(search_window)
        peak_idx_in_window = np.argmax(search_window)
        true_lag = min_lag + peak_idx_in_window
        
        # 估算频率
        est_freq = self.fps / true_lag if true_lag > 0 else 0.0
        
        # 4. 判定
        # 峰值必须足够高 (ACF > Threshold)
        # 且必须是局部峰值 (比周围的点都高，避免单调下降的误判)
        is_peak = True
        if true_lag > 0 and true_lag < len(r) - 1:
            if r[true_lag] < r[true_lag-1] or r[true_lag] < r[true_lag+1]:
                is_peak = False
        
        if is_peak and peak_val > self.ACF_THRESHOLD and is_amplitude_sufficient:
            return True, est_freq, peak_val
        else:
            return False, est_freq, peak_val


class MouseTracker:
    """
    小鼠追踪器（基于ROI）
    每个ROI框内的检测自动关联为同一只老鼠
    
    注：所有速度/光流值均为归一化值（基于6cm标准体长），非真实物理单位
    """
    
    def __init__(self, 
                 roi_polygon,
                 roi_id,
                 bg_subtractor,
                 morphology_processor,
                 detector,
                 flow_analyzer,
                 use_yolo_seg: bool = False,
                 yolo_model_path: Optional[str] = None,
                 yolo_conf: float = 0.25,
                 yolo_imgsz: int = 640,
                 fps=30.0,
                 pixel_to_mm=1.0,
                 roi_name=None,
                 time_config=None,
                 # 判定参数（黄金参数 - 基于6cm标准体长归一化）
                 win_seconds=0.3,  # 滑动窗口时长（秒）
                 flow_thr_hi=50.0,  # 光流高阈值（归一化值）
                 flow_thr_lo=35.0,  # 光流低阈值（归一化值）
                 speed_std_thr_hi=40.0,  # 质心速度高阈值（归一化值）
                 speed_std_thr_lo=20.0,  # 质心速度低阈值（归一化值）
                 min_active_seconds=0.2,  # 最小激活时长（秒）
                 calm_seconds=0.5,  # 冷却时长（秒）
                 coherence_thr=0.8,  # 方向一致性阈值
                 coherence_calm_seconds=0.5):  # 方向一致性判定时长（秒，默认0.5s）
        """
        初始化追踪器
        
        参数:
            roi_polygon: ROI多边形顶点 [(x1,y1), (x2,y2), ...]
            roi_id: ROI编号
            bg_subtractor: 背景减除器
            morphology_processor: 形态学处理器
            detector: 检测器
            flow_analyzer: 光流分析器
            fps: 帧率
            pixel_to_mm: 像素到毫米转换比例
            roi_name: ROI名称
            time_config: 时间配置 {mode: 'global'|'relative'|'precise', start: float, end: float}
            
            # 行为判定参数（黄金参数 - 基于6cm标准体长归一化）
            win_seconds: 滑动窗口时长（秒，默认0.3s）
            flow_thr_hi: 光流高阈值（归一化值，默认50）
            flow_thr_lo: 光流低阈值（归一化值，默认35）
            speed_std_thr_hi: 质心速度标准差高阈值（归一化值，默认40）
            speed_std_thr_lo: 质心速度标准差低阈值（归一化值，默认20）
            min_active_seconds: 最小激活时长（秒，默认0.2s）
            calm_seconds: 冷却时长（秒，默认0.4s）
            coherence_thr: 方向一致性阈值（无量纲，默认0.8）
            coherence_calm_seconds: 方向一致性判定时长（秒，默认0.5s）
        """
        self.roi_polygon = np.array(roi_polygon, dtype=np.int32)
        self.roi_id = roi_id
        self.roi_name = roi_name if roi_name else f"ROI {roi_id}"
        self.time_config = time_config
        # 确保时间配置参数为浮点数
        if self.time_config:
            if self.time_config.get('start') is not None:
                try:
                    self.time_config['start'] = float(self.time_config['start'])
                except:
                    pass
            if self.time_config.get('end') is not None:
                try:
                    self.time_config['end'] = float(self.time_config['end'])
                except:
                    pass
                    
        self.bg_subtractor = bg_subtractor
        self.morphology_processor = morphology_processor
        self.detector = detector
        self.flow_analyzer = flow_analyzer

        # YOLO-seg 检测（可选）：用于替代 bg_subtractor + contour detector
        self.use_yolo_seg = bool(use_yolo_seg)
        self.yolo_model_path = str(yolo_model_path) if yolo_model_path else None
        self.yolo_conf = float(yolo_conf)
        self.yolo_imgsz = int(yolo_imgsz)
        self.yolo_detector: Optional[YoloSegDetector] = None
        if self.use_yolo_seg and self.yolo_model_path:
            try:
                self.yolo_detector = YoloSegDetector(self.yolo_model_path, conf=self.yolo_conf, imgsz=self.yolo_imgsz)
            except Exception as e:
                # 若依赖/模型不可用，则回退到传统方法
                self.yolo_detector = None
                self.use_yolo_seg = False
                print(f"⚠ YOLO分割检测初始化失败，将回退到背景减除: {e}")
        
        # 物理参数
        self.fps = fps
        self.pixel_to_mm = pixel_to_mm
        
        # 将时间参数转换为帧数
        self.win = max(1, int(win_seconds * fps))  # 0.3秒 * 30fps = 9帧
        self.min_frames_active = max(1, int(min_active_seconds * fps))  # 0.2秒 * 30fps = 6帧
        self.calm_frames = max(1, int(calm_seconds * fps))  # 0.5秒 * 30fps = 15帧
        self.coherence_calm_frames = max(1, int(coherence_calm_seconds * fps))  # 0.5秒 * 30fps = 15帧
        
        # 判定阈值（物理单位）
        self.flow_thr_hi = flow_thr_hi
        self.flow_thr_lo = flow_thr_lo
        self.speed_std_thr_hi = speed_std_thr_hi
        self.speed_std_thr_lo = speed_std_thr_lo
        self.coherence_thr = coherence_thr
        
        # 创建ROI掩码
        self.roi_mask = None
        self.roi_bbox = None  # (x, y, w, h)
        
        # 追踪状态
        self.trajectory = deque(maxlen=100)  # 质心轨迹
        self.last_centroid = None  # 上一帧质心
        self.first_detected_time = None # 首次检测到老鼠的时间（秒）
        
        # 时序特征
        self.speeds = deque(maxlen=self.win)  # 速度序列（归一化值）
        self.flows = deque(maxlen=self.win)  # 光流序列（归一化值）
        self.coherences = deque(maxlen=self.coherence_calm_frames)  # 方向一致性序列（只需要0.5秒）

        # 加速度/jerk（用于区分“整体摆动”与“局部乱动”）
        accel_win = max(self.win, int(0.5 * fps))
        self._prev_speed = None
        self.accels = deque(maxlen=accel_win)
        
        # 判定状态
        self.is_struggling = False
        self.frames_active = 0
        self.calm_counter = 0
        self.frame_count = 0
        self.detected_count = 0  # 检测到老鼠的帧数
        
        # 不动时间统计
        self.immobility_stats = {
            'first_frame': 0,
            'total_frames': 0,
            'immobility_frames': 0,
            'struggling_frames': 0,
            'immobility_episodes': 0,
            'struggling_episodes': 0,
            'latency_to_immobility': None,
            'last_state': None,
        }
        """
        初始化追踪器
        
        参数:
            roi_polygon: ROI多边形顶点 [(x1,y1), (x2,y2), ...]
            roi_id: ROI编号
            bg_subtractor: 背景减除器
            morphology_processor: 形态学处理器
            detector: 检测器
            flow_analyzer: 光流分析器
            fps: 帧率
            pixel_to_mm: 像素到毫米转换比例
            
            # 行为判定参数（黄金参数 - 基于6cm标准体长归一化）
            win_seconds: 滑动窗口时长（秒，默认0.3s）
            flow_thr_hi: 光流高阈值（归一化值，默认50）
            flow_thr_lo: 光流低阈值（归一化值，默认35）
            speed_std_thr_hi: 质心速度标准差高阈值（归一化值，默认40）
            speed_std_thr_lo: 质心速度标准差低阈值（归一化值，默认20）
            min_active_seconds: 最小激活时长（秒，默认0.2s）
            calm_seconds: 冷却时长（秒，默认0.4s）
            coherence_thr: 方向一致性阈值（无量纲，默认0.8）
            coherence_calm_seconds: 方向一致性判定时长（秒，默认0.5s）
        """
        self.roi_polygon = np.array(roi_polygon, dtype=np.int32)
        self.roi_id = roi_id
        self.bg_subtractor = bg_subtractor
        self.morphology_processor = morphology_processor
        self.detector = detector
        self.flow_analyzer = flow_analyzer
        
        # 物理参数
        self.fps = fps
        self.pixel_to_mm = pixel_to_mm
        
        # 将时间参数转换为帧数
        self.win = max(1, int(win_seconds * fps))  # 0.3秒 * 30fps = 9帧
        self.min_frames_active = max(1, int(min_active_seconds * fps))  # 0.2秒 * 30fps = 6帧
        self.calm_frames = max(1, int(calm_seconds * fps))  # 0.5秒 * 30fps = 15帧
        self.coherence_calm_frames = max(1, int(coherence_calm_seconds * fps))  # 0.5秒 * 30fps = 15帧
        
        # 判定阈值（物理单位）
        self.flow_thr_hi = flow_thr_hi
        self.flow_thr_lo = flow_thr_lo
        self.speed_std_thr_hi = speed_std_thr_hi
        self.speed_std_thr_lo = speed_std_thr_lo
        self.coherence_thr = coherence_thr
        
        # 创建ROI掩码
        self.roi_mask = None
        self.roi_bbox = None  # (x, y, w, h)
        
        # 追踪状态
        self.trajectory = deque(maxlen=100)  # 质心轨迹
        self.last_centroid = None  # 上一帧质心
        
        # 时序特征
        self.speeds = deque(maxlen=self.win)  # 速度序列（归一化值）
        self.flows = deque(maxlen=self.win)  # 光流序列（归一化值）
        self.coherences = deque(maxlen=self.coherence_calm_frames)  # 方向一致性序列（只需要0.5秒）
        
        # 判定状态
        self.is_struggling = False
        self.frames_active = 0
        self.calm_counter = 0
        self.frame_count = 0
        self.detected_count = 0  # 检测到老鼠的帧数
        
        # 不动时间统计
        self.immobility_stats = {
            'first_frame': 0,
            'total_frames': 0,
            'immobility_frames': 0,
            'struggling_frames': 0,
            'immobility_episodes': 0,
            'struggling_episodes': 0,
            'latency_to_immobility': None,
            'last_state': None,
        }
        
        # 周期性分析器
        self.periodicity_analyzer = PeriodicityAnalyzer(fps)

        # 自适应基线：用于“近景/视角变化”时的形态阈值与尺度归一化
        baseline_len = max(5, int(2.0 * fps))
        self._bbox_h_history = deque(maxlen=baseline_len)
        self._aspect_ratio_history = deque(maxlen=baseline_len)
        self._baseline_ready_frames = max(5, int(1.0 * fps))
        
        # 摆动判定状态机
        self.is_in_confirmed_swing = False # 是否处于确认的摆动状态
        self.min_swing_frames = int(1.5 * fps) # 摆动确认窗口（约1.5秒）
        # 说明：旧实现要求“连续满足条件 >= min_swing_frames”，在摆动端点/短暂遮挡时会频繁清零。
        # 新实现用滑窗证据（命中数）做迟滞确认，更符合“摆动是连续行为”的物理直觉。
        self.swing_buffer = deque(maxlen=self.min_swing_frames)  # [{'is_struggling': bool, 'swing_candidate': bool}, ...]
        self.swing_evidence = deque(maxlen=self.min_swing_frames)  # 0/1
        self.swing_confirm_hits = max(1, int(0.65 * self.min_swing_frames))
        self.swing_release_hits = max(0, int(0.35 * self.min_swing_frames))
        
        # 旧变量保留（兼容性），但不再主要使用
        self.swing_persistence = 0
        self.swing_duration_counter = 0
    
    def initialize_roi_mask(self, frame_shape):
        """初始化ROI掩码"""
        self.roi_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_polygon], 255)
        
        # 计算ROI边界框
        x, y, w, h = cv2.boundingRect(self.roi_polygon)
        self.roi_bbox = (x, y, w, h)
    
    def update(self, frame, current_time=None, gray=None, yolo_dets_full=None):
        """
        更新追踪（处理一帧）
        
        参数:
            frame: 当前帧
            current_time: 当前视频时间（秒），用于精确时间控制
            gray: 预计算的灰度图（可选，避免重复转换）
        
        返回:
            Dict: 追踪结果
        """
        # 性能分析计时
        import time
        if not hasattr(self, '_perf_times'):
            self._perf_times = {'roi': [], 'bg_sub': [], 'morph': [], 'detect': [], 'flow': [], 'stats': []}
        
        if self.roi_mask is None:
            self.initialize_roi_mask(frame.shape)
        
        self.frame_count += 1
        
        # 【性能优化】预计算灰度图
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 【性能优化】使用 bounding box 裁剪而不是全图 bitwise_and
        t0 = time.time()
        x, y, w, h = self.roi_bbox
        
        # 裁剪 ROI 区域（比 bitwise_and 快很多）
        roi_frame = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        
        # 裁剪后的 ROI 掩码
        if not hasattr(self, '_cropped_roi_mask') or self._cropped_roi_mask is None:
            self._cropped_roi_mask = self.roi_mask[y:y+h, x:x+w]
        
        self._perf_times['roi'].append(time.time() - t0)
        
        # 2-4. 检测小鼠（YOLO分割优先；否则回退到背景减除+轮廓检测）
        fg_mask = None
        detections = []

        if self.use_yolo_seg and self.yolo_detector is not None:
            # YOLO 分割：在全帧上推理（避免ROI裁剪导致尺度分布变化），再把检测归属到本ROI
            t0 = time.time()
            if yolo_dets_full is None:
                yolo_dets_full = self.yolo_detector.detect(frame)
            self._perf_times['detect'].append(time.time() - t0)

            best = None
            best_score = 0.0

            # 优先使用“质心在ROI内”的检测；若没有，则回退到“mask与ROI bbox 区域有交集”的检测
            candidates_in = []
            candidates_overlap = []
            for d in (yolo_dets_full or []):
                mask_full = d.get('mask')
                if mask_full is None or not isinstance(mask_full, np.ndarray):
                    continue
                if mask_full.shape[:2] != frame.shape[:2]:
                    continue

                cx, cy = d.get('centroid', (None, None))
                if cx is None or cy is None:
                    continue
                try:
                    inside = cv2.pointPolygonTest(self.roi_polygon, (float(cx), float(cy)), False) >= 0
                except Exception:
                    inside = False

                # overlap：只在 ROI bbox 局部计算（避免全帧 bool 运算导致极慢）
                try:
                    mask_roi = mask_full[y:y+h, x:x+w]
                    overlap = cv2.bitwise_and(mask_roi, self._cropped_roi_mask)
                    overlap_area = int(np.count_nonzero(overlap))
                except Exception:
                    overlap_area = 0

                if inside:
                    candidates_in.append((d, overlap_area))
                elif overlap_area > 0:
                    candidates_overlap.append((d, overlap_area))

            candidates = candidates_in if len(candidates_in) > 0 else candidates_overlap
            for d, overlap_area in candidates:
                conf = float(d.get('conf', 0.0))
                score = conf * float(overlap_area)
                if score > best_score:
                    best_score = score
                    best = (d, overlap_area)

            if best is not None:
                d, overlap_area = best
                mask_full = d['mask']

                # 生成 ROI-local fg_mask（用于光流统计），并强制限制在ROI内
                fg_mask = mask_full[y:y+h, x:x+w].copy()
                fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=self._cropped_roi_mask)

                # bbox/centroid 使用全局坐标（来自全帧推理）
                detections = [
                    {
                        'centroid': d.get('centroid'),
                        'bbox': d.get('bbox'),
                        'area': int(overlap_area),
                        'contour': None,
                        'conf': d.get('conf', 0.0),
                        '_yolo_full': True,
                    }
                ]
            else:
                fg_mask = np.zeros((roi_frame.shape[0], roi_frame.shape[1]), dtype=np.uint8)
        else:
            # 传统方法：背景减除 -> ROI裁剪掩码 -> 形态学 -> 轮廓检测
            t0 = time.time()
            fg_mask_raw = self.bg_subtractor.apply(roi_frame, gray=roi_gray)
            fg_mask = cv2.bitwise_and(fg_mask_raw, fg_mask_raw, mask=self._cropped_roi_mask)
            self._perf_times['bg_sub'].append(time.time() - t0)

            t0 = time.time()
            fg_mask = self.morphology_processor.process(fg_mask, inplace=True)
            self._perf_times['morph'].append(time.time() - t0)

            t0 = time.time()
            detections = self.detector.detect(fg_mask)
            self._perf_times['detect'].append(time.time() - t0)
        
        if len(detections) == 0:
            # 即使未检测到，也视为不动（如果已经在追踪中）
            # 确保统计数据更新，否则报告会为空
            # 注意：这里不增加frames_active，因为没有检测到活动
            # 但是我们需要更新时间统计（如果已经在时间段内）
            self._update_immobility_stats(False, current_time)
            
            return {
                'roi_id': self.roi_id,
                'roi_name': self.roi_name,
                'detected': False,
                'centroid': None,
                'contour': None,
                'flow': None,
                'is_struggling': False,
                'fg_mask': fg_mask,
                'roi_offset': (x, y)  # 记录偏移量
            }
        
        # 选择最大的检测（假设是小鼠）
        detection = max(detections, key=lambda d: d.get('area', 0))

        # 【重要】统一 detection 坐标系：
        # - 传统路径：检测在 ROI-crop 上完成，需要加 (x,y) 偏移到全局
        # - YOLO全帧路径：检测已经是全局坐标，不做偏移
        if detection.get('_yolo_full'):
            # 确保 centroid / bbox 形状正确
            pass
        else:
            # centroid 偏移
            local_centroid = detection['centroid']
            global_centroid = (local_centroid[0] + x, local_centroid[1] + y)

            # bbox 偏移
            local_bbox = detection['bbox']  # (bx, by, bw, bh)
            global_bbox = (local_bbox[0] + x, local_bbox[1] + y, local_bbox[2], local_bbox[3])

            # contour 偏移（YOLO分割路径可能没有轮廓）
            local_contour = detection.get('contour')
            global_contour = None
            if local_contour is not None:
                global_contour = local_contour + np.array([[x, y]])

            # 更新 detection 为全局坐标
            detection['centroid'] = global_centroid
            detection['bbox'] = global_bbox
            detection['contour'] = global_contour
        
        # 记录首次检测时间
        if self.first_detected_time is None:
            self.first_detected_time = current_time if current_time is not None else (self.frame_count / self.fps)
        
        # 5. 计算速度（mm/s）
        centroid = detection['centroid']
        if self.last_centroid is None:
            speed = 0.0
        else:
            dist_px = np.linalg.norm(np.array(centroid) - np.array(self.last_centroid))
            dist_mm = dist_px * self.pixel_to_mm
            speed = dist_mm * self.fps  # 归一化速度值
        self.last_centroid = centroid
        
        # 6. 光流分析（使用预计算灰度图和边界框）
        t0 = time.time()
        # 从检测结果获取边界框（全局坐标）
        bbox = detection['bbox']  # (bx, by, bw, bh) - 已是全局坐标
        margin = 10
        flow_bbox = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(frame.shape[1], bbox[0] + bbox[2] + margin),
            min(frame.shape[0], bbox[1] + bbox[3] + margin)
        )
        # 传递 mask_offset，因为 fg_mask 是裁剪后的局部 mask
        flow_result = self.flow_analyzer.analyze(
            frame, fg_mask, 
            curr_gray=gray, 
            bbox=flow_bbox, 
            mask_offset=(x, y)  # x, y 是 ROI 的 bbox 偏移
        )
        self._perf_times['flow'].append(time.time() - t0)
        
        if flow_result is None:
            # 第一帧，无法计算光流
            flow_mm_s = 0.0
            coherence = 0.0
        else:
            flow_mm_s = flow_result['flow_p90_mm_s']  # 使用P90（更稳健）
            coherence = flow_result['coherence']
        
        # 7. 更新时序特征
        t0 = time.time()
        self.speeds.append(speed)
        self.flows.append(flow_mm_s)
        self.coherences.append(coherence)

        # 加速度（mm/s^2）
        if self._prev_speed is None:
            accel = 0.0
        else:
            accel = (speed - self._prev_speed) * self.fps
        self._prev_speed = speed
        self.accels.append(float(accel))
        
        # 8. 计算统计量
        speed_arr = np.array(self.speeds)
        flow_arr = np.array(self.flows)
        coherence_arr = np.array(self.coherences)
        accel_arr = np.array(self.accels) if len(self.accels) > 0 else np.array([0.0])
        
        speed_std = float(np.std(speed_arr)) if len(speed_arr) >= 3 else 0.0
        flow_p90 = float(np.percentile(flow_arr, 90)) if len(flow_arr) >= 3 else flow_mm_s
        accel_std = float(np.std(accel_arr)) if len(accel_arr) >= 3 else 0.0
        self._perf_times['stats'].append(time.time() - t0)
        
        # 9. 复杂判定逻辑（引入形态学 + 熵的科学判定）
        self.frames_active += 1
        
        # A. 周期性检测 (基于质心轨迹，使用相对尺度避免近景幅度失真)
        cx = detection['centroid'][0]
        cy = detection['centroid'][1]
        self.periodicity_analyzer.update(cx, cy)
        # 使用 ROI 内尺度（bbox 高度中位数）做相对幅度检测，避免“拍得近→幅度很大”导致阈值失真
        self._bbox_h_history.append(float(h))
        scale_px = float(np.median(self._bbox_h_history)) if len(self._bbox_h_history) > 0 else float(h)
        is_periodic, swing_freq, acf_peak = self.periodicity_analyzer.is_swinging_relative(
            scale_px=scale_px,
            # 近景/ROI抖动下，amp_ratio 往往在 0.012~0.03 波动；过高会导致周期性检测永远不“通过”。
            min_amplitude_ratio=0.012
        )

        # 额外：相对幅度（用于调试）
        amp_ratio = 0.0
        y_over_x = 0.0
        try:
            if len(self.periodicity_analyzer.x_history) > 3 and scale_px > 1e-6:
                amp_ratio = float(np.std(np.array(self.periodicity_analyzer.x_history, dtype=np.float32)) / scale_px)
            # 关键：挣扎时往往伴随更强的纵向(上下)扰动；摆动则以横向左右为主。
            _, _, yx = self.periodicity_analyzer.motion_anisotropy()
            if yx is not None:
                y_over_x = float(yx)
        except Exception:
            amp_ratio = 0.0
            y_over_x = 0.0
        
        # B. 形态学检测 (Morphology Check) - 关键！
        # 挣扎时老鼠会卷曲 (Curling)，导致 高宽比(AspectRatio) 变小，或 边界框高度(Height) 变小
        # 摆动时老鼠被重力拉长，身体应该是细长的
        _, _, w, h = detection['bbox']
        aspect_ratio = h / (w + 1e-6) # 高宽比
        
        # 设定一个形态阈值（自适应基线）：
        # 近景/角度变化会让固定阈值(如1.8)失效，因此用最近2秒的中位数作为基线。
        self._aspect_ratio_history.append(float(aspect_ratio))
        baseline_ar = None
        if len(self._aspect_ratio_history) >= self._baseline_ready_frames:
            baseline_ar = float(np.median(self._aspect_ratio_history))

        if baseline_ar is None:
            is_elongated = aspect_ratio > 1.8
        else:
            # baseline_ar*0.75：允许一定形变，但仍倾向“细长”；同时设置下限避免阈值过低
            is_elongated = aspect_ratio > max(1.2, baseline_ar * 0.75)
        
        # C. 运动一致性检测 (Coherence Check) - 仅作参考
        # 用户反馈：Coherence 在摆动端点也会骤降（即使速度不为0），导致判定中断。
        # 因此不再将其作为硬性指标，仅用于调试观察。
        coherence = flow_result.get('coherence', 0.0) if flow_result else 0.0
        flow_entropy = flow_result.get('flow_entropy', 1.0) if flow_result else 1.0
        
        # D. 打分式判定（更鲁棒）
        # 直觉：
        # - 摆动：强周期性 + 低熵(整体同向运动) + 高一致性 + “内部运动/平移”不夸张
        # - 挣扎：高熵(多方向) + 低一致性 + 高加速度波动/内部运动偏大
        eps = 1e-6
        flow_over_speed = float(flow_p90 / (abs(speed) + eps))

        coh_med = float(np.median(coherence_arr)) if len(coherence_arr) > 0 else float(coherence)
        ent = float(flow_entropy)

        swing_score = 0.0
        struggle_score = 0.0

        # 周期性：权重最高
        if acf_peak >= 0.80:
            swing_score += 2.5
        elif acf_peak >= 0.70:
            swing_score += 2.0
        elif acf_peak >= 0.60:
            swing_score += 1.0

        # 频率落在摆动区间（不是必须，但能减少噪声）
        if 0.8 <= swing_freq <= 3.0:
            swing_score += 0.8

        # 整体同向：高一致性 + 低熵
        if coh_med >= 0.90:
            swing_score += 1.2
        elif coh_med >= 0.80:
            swing_score += 0.8
        else:
            struggle_score += 0.6

        if ent <= 0.15:
            swing_score += 1.2
        elif ent <= 0.30:
            swing_score += 0.8
        elif ent >= 0.55:
            struggle_score += 1.5
        elif ent >= 0.40:
            struggle_score += 0.8

        # 形态：弱约束（近景/视角下不强依赖）
        if is_elongated:
            swing_score += 0.6
        else:
            struggle_score += 0.4

        # “内部运动/平移比”：挣扎往往 > 摆动（摆动更像刚体平移）
        if flow_over_speed >= 3.0:
            struggle_score += 1.2
        elif flow_over_speed >= 2.0:
            struggle_score += 0.6
        elif flow_over_speed <= 1.2:
            swing_score += 0.4

        # 加速度波动：挣扎时 jerk/accel 波动更大（但摆动端点也会抖，所以权重适中）
        if accel_std >= 150.0 and ent >= 0.35:
            struggle_score += 0.8

        # 最终：摆动候选（硬门控 + 去抖动）
        # 关键点：不要让端点的短暂“熵升高/一致性下降”把整个摆动段打断清零。
        # 因此这里用 ACF/频率/相对幅度 做硬门控，熵/一致性只作为门槛而非强制拉开分差。
        # 反误判关键：有些“挣扎”也会让质心呈左右周期，但其内部光流更乱（高熵/低一致性）或
        # “内部运动/平移比”明显更大（flow_over_speed 高）。因此摆动候选必须同时满足“刚体式摆动”的门控。
        freq_ok = 0.8 <= swing_freq <= 3.0
        acf_ok = acf_peak >= 0.45
        amp_ok = amp_ratio >= 0.012

        # 反误判关键（ROI3 15-20s）：挣扎会产生较大的纵向扰动，导致 std(y)/std(x) 偏高。
        # 对“左右摆动”来说，y_over_x 通常显著更低。
        yx_ok = (y_over_x <= 0.55) if (y_over_x is not None) else True

        # 更严格的“刚体摆动”门控
        coh_ok = coh_med >= 0.88
        ent_ok = ent <= 0.30

        # flow_over_speed 在 speed→0 时会被放大：端点允许豁免该门控（靠滑窗证据吸收少量失败帧）
        speed_floor_for_fos = 8.0  # mm/s
        fos_ok = True
        if abs(speed) >= speed_floor_for_fos:
            fos_ok = flow_over_speed <= 2.2

        # “明显挣扎”反门控：即便周期性强，只要内部运动很乱，也不要计为摆动证据
        struggle_like = (ent >= 0.55) or (coh_med <= 0.75) or (abs(speed) >= speed_floor_for_fos and flow_over_speed >= 3.0)

        is_swinging_condition = bool(freq_ok and acf_ok and amp_ok and yx_ok and coh_ok and ent_ok and fos_ok and (not struggle_like))
        
        # E. 状态更新 (带迟滞/去抖动 + 最小持续时间过滤)
        # 逻辑：摆动必须持续至少1.5秒才被确认。
        # 一旦确认，之前的1.5秒也应该被视为摆动（统计修正）。
        
        # 将当前帧写入证据/缓冲（用于回溯修正）
        # 注意：这里先记录 is_struggling=False 占位，后面会用 final_struggling 回填。
        if not self.is_in_confirmed_swing:
            self.swing_buffer.append({'is_struggling': False, 'swing_candidate': bool(is_swinging_condition)})
        self.swing_evidence.append(1 if is_swinging_condition else 0)

        # 确认/退出逻辑（滑窗迟滞）
        if not self.is_in_confirmed_swing:
            if len(self.swing_evidence) >= self.min_swing_frames:
                if int(sum(self.swing_evidence)) >= int(self.swing_confirm_hits):
                    self.is_in_confirmed_swing = True

                    # 【关键】回溯修正统计数据
                    # 仅修正 swing_candidate=True 的帧，避免把真正挣扎段“洗掉”。
                    for frame_info in self.swing_buffer:
                        if frame_info.get('swing_candidate') and frame_info.get('is_struggling'):
                            self.immobility_stats['struggling_frames'] -= 1
                            self.immobility_stats['immobility_frames'] += 1

                    self.swing_buffer.clear()
                    self.swing_evidence.clear()
                    print(f"[ROI {self.roi_id}] Swing Confirmed! Retroactively fixed stats.")
        else:
            # 已确认：当证据持续不足时退出
            if len(self.swing_evidence) >= self.min_swing_frames:
                if int(sum(self.swing_evidence)) <= int(self.swing_release_hits):
                    self.is_in_confirmed_swing = False
                    self.swing_evidence.clear()
            
        # 最终生效的摆动状态
        is_swinging_state = self.is_in_confirmed_swing

        # [DEBUG] 打印关键指标，帮助用户调试阈值
        if self.frame_count % 30 == 0 and detection:
            base_txt = f"baseAR={baseline_ar:.2f}" if baseline_ar is not None else "baseAR=N/A"
            print(
                f"[ROI {self.roi_id}] Swing/Struggle: swingScore={swing_score:.2f}, struggleScore={struggle_score:.2f}, "
                f"AR={aspect_ratio:.2f} ({base_txt}), AmpR={amp_ratio:.3f}, Hmed={scale_px:.1f}px, "
                f"Y/X={y_over_x:.2f}, "
                f"CohMed={coh_med:.2f}, Ent={ent:.2f}, Flow={flow_p90:.1f}, Speed={speed:.1f}, "
                f"FOS={flow_over_speed:.2f}, AccStd={accel_std:.1f}, Freq={swing_freq:.2f}Hz, ACF={acf_peak:.2f}, "
                f"SwingState={is_swinging_state}, Buffer={int(sum(self.swing_evidence))}/{self.min_swing_frames}"
            )

        # F. 最终挣扎判定
        if is_swinging_state:
            # 确认为摆动 -> 强制不动
            self.is_struggling = False
            self.calm_counter = 0
            self.swing_persistence = self.fps * 0.2
        else:
            # 非摆动状态 -> 正常判定
            if self.swing_persistence > 0:
                self.swing_persistence -= 1
                self.is_struggling = False
            else:
                # 挣扎判定逻辑（稳定版）：保持原有双阈值 + 迟滞
                # 说明：之前引入的 flow_over_speed/entropy 触发在 speed≈0 时会放大噪声，
                # 导致大量静止帧被误判为挣扎，因此这里回退为更可靠的基础判定。
                was = self.is_struggling

                if not was:
                    # 当前不动：使用高阈值进入挣扎
                    # 经验问题：视频开头/光照自适应/背景模型未稳时，可能出现“整体平移式”的光流尖峰
                    # （coherence很高、entropy很低、flow_over_speed接近1），即便老鼠实际上不动也会误触发。
                    # 因此：速度波动仍可直接触发；但仅靠 flow_p90 触发时，要求 flow_over_speed 足够大（更像内部乱动）。
                    enter_by_speed = speed_std > self.speed_std_thr_hi
                    enter_by_flow = (flow_p90 > self.flow_thr_hi and flow_over_speed >= 2.5)
                    if enter_by_speed or enter_by_flow:
                        self.is_struggling = True
                        self.calm_counter = 0
                else:
                    # 当前挣扎：使用低阈值判断退出
                    if flow_p90 < self.flow_thr_lo and speed_std < self.speed_std_thr_lo:
                        self.calm_counter += 1
                    else:
                        self.calm_counter = 0

                    # 连续 calm_frames 帧都低于阈值才退出
                    if self.calm_counter >= self.calm_frames:
                        self.is_struggling = False
                        self.calm_counter = 0
        
        # 最终判定（需要足够激活帧数）
        final_struggling = bool(self.is_struggling if self.frames_active >= self.min_frames_active else False)
        
        # 【关键】更新缓冲区中的挣扎状态
        # 如果当前处于潜在摆动期（swing_buffer不为空），我们需要记录这一帧最终被判定为什么
        # 以便将来如果确认为摆动，可以回溯修正
        if len(self.swing_buffer) > 0:
            self.swing_buffer[-1]['is_struggling'] = final_struggling
        
        # 10. 更新统计
        self._update_immobility_stats(final_struggling, current_time)
        
        # 11. 记录轨迹
        self.trajectory.append(detection['centroid'])
        self.detected_count += 1  # 检测成功计数
        
        # 【关键修复】可视化状态同步
        # 如果处于抑制期，也应该显示为Swinging，否则用户会困惑为什么判为不动
        effective_is_swinging = is_swinging_state or (self.swing_persistence > 0)

        return {
            'roi_id': self.roi_id,
            'roi_name': self.roi_name,
            'detected': True,
            'centroid': detection['centroid'],
            'contour': detection['contour'],
            'area': detection['area'],
            'bbox': detection['bbox'],
            'flow': flow_result,
            'is_struggling': final_struggling,
            'fg_mask': fg_mask,
            'roi_offset': (0, 0),
            'trajectory': list(self.trajectory),
            # 判定参数（显示用）
            'speed': speed,
            'speed_std': speed_std,
            'flow_p90': flow_p90,
            'coherence': coherence,  # 当前帧方向一致性
            'is_swinging': effective_is_swinging, # 返回生效的状态，而不仅仅是瞬时检测
            'calm_counter': self.calm_counter,
            'aspect_ratio': aspect_ratio,
            'flow_entropy': float(flow_entropy),
            'swing_freq': swing_freq,
            'acf_peak': acf_peak,
            'swing_conf': int(sum(self.swing_evidence)) if not self.is_in_confirmed_swing else self.min_swing_frames, # 导出内部计数器供可视化
            'amp_ratio': float(amp_ratio),
            'y_over_x': float(y_over_x),
            'flow_over_speed': float(flow_over_speed),
            'accel_std': float(accel_std),
        }
    
    def _update_immobility_stats(self, is_struggling, current_time=None):
        """更新不动时间统计"""
        # 检查时间配置
        if self.time_config:
            if current_time is None:
                # 如果没有提供current_time，尝试使用内部估算
                current_time = self.frame_count / self.fps
            
            mode = self.time_config.get('mode', 'global')
            start_cfg = self.time_config.get('start')
            end_cfg = self.time_config.get('end')
            
            if mode == 'precise':
                # 精确时间段（秒）
                if start_cfg is not None and current_time < start_cfg:
                    return
                if end_cfg is not None and current_time > end_cfg:
                    return
            elif mode == 'relative':
                # 相对检测时间（秒）
                if self.first_detected_time is None:
                    return # 还没检测到，不统计
                
                elapsed = current_time - self.first_detected_time
                if start_cfg is not None and elapsed < start_cfg:
                    return
                if end_cfg is not None and elapsed > end_cfg:
                    return

        stats = self.immobility_stats
        
        # 设置第一帧
        if stats['first_frame'] == 0:
            stats['first_frame'] = self.frame_count
        
        stats['total_frames'] += 1
        
        # 只要在时间段内，就应该统计（无论是否active）
        # 如果未激活（frames_active < min），默认为不动
        # 或者我们可以保留原逻辑：只在激活后统计？
        # 用户反馈数据为0，说明可能一直没激活。
        # 为了保证有数据，我们移除 frames_active 的限制，或者默认视为不动
        
        if is_struggling and self.frames_active >= self.min_frames_active:
            stats['struggling_frames'] += 1
        else:
            # 不动（包括未激活状态）
            stats['immobility_frames'] += 1
        
        # 检测状态切换
        if stats['last_state'] is not None and stats['last_state'] != is_struggling:
            if is_struggling and self.frames_active >= self.min_frames_active:
                stats['struggling_episodes'] += 1
            elif not is_struggling:
                stats['immobility_episodes'] += 1
                if stats['latency_to_immobility'] is None:
                    stats['latency_to_immobility'] = self.frame_count - stats['first_frame']
        
        stats['last_state'] = is_struggling
    
    def get_immobility_summary(self, fps):
        """
        获取不动时间统计摘要
        
        参数:
            fps: 帧率
        
        返回:
            Dict: 统计摘要
        """
        stats = self.immobility_stats
        
        if stats['total_frames'] == 0:
            # 即使没有统计到帧（例如在指定时间段内未检测到），也返回全0结果，确保报告生成
            return {
                'roi_id': self.roi_id,
                'roi_name': self.roi_name,
                'time_config': self.time_config,
                'total_time_seconds': 0.0,
                'immobility_time_seconds': 0.0,
                'struggling_time_seconds': 0.0,
                'immobility_percent': 0.0,
                'struggling_percent': 0.0,
                'immobility_episodes': 0,
                'struggling_episodes': 0,
                'immobility_frequency_per_min': 0.0,
                'struggling_frequency_per_min': 0.0,
                'latency_to_immobility_seconds': None,
                'total_frames': 0,
            }
        
        total_time = stats['total_frames'] / fps
        immobile_time = stats['immobility_frames'] / fps
        struggling_time = stats['struggling_frames'] / fps
        
        immobile_percent = (stats['immobility_frames'] / stats['total_frames']) * 100
        struggling_percent = (stats['struggling_frames'] / stats['total_frames']) * 100
        
        duration_minutes = total_time / 60.0
        immobile_freq = stats['immobility_episodes'] / max(duration_minutes, 1/60.0)
        struggling_freq = stats['struggling_episodes'] / max(duration_minutes, 1/60.0)
        
        latency_seconds = None
        if stats['latency_to_immobility'] is not None:
            latency_seconds = stats['latency_to_immobility'] / fps
        
        return {
            'roi_id': self.roi_id,
            'roi_name': self.roi_name,
            'time_config': self.time_config,  # 添加时间配置信息
            'total_time_seconds': round(total_time, 2),
            'immobility_time_seconds': round(immobile_time, 2),
            'struggling_time_seconds': round(struggling_time, 2),
            'immobility_percent': round(immobile_percent, 1),
            'struggling_percent': round(struggling_percent, 1),
            'immobility_episodes': stats['immobility_episodes'],
            'struggling_episodes': stats['struggling_episodes'],
            'immobility_frequency_per_min': round(immobile_freq, 2),
            'struggling_frequency_per_min': round(struggling_freq, 2),
            'latency_to_immobility_seconds': round(latency_seconds, 2) if latency_seconds is not None else None,
            'total_frames': stats['total_frames'],
        }


def draw_results(frame, tracker_results, show_trajectory=True, show_flow_vectors=False, inplace=False):
    """
    在帧上绘制追踪结果
    
    参数:
        frame: 输入帧
        tracker_results: 追踪器结果列表
        show_trajectory: 是否显示轨迹
        show_flow_vectors: 是否显示光流向量
        inplace: 是否原地修改（避免复制，更快）
    
    返回:
        绘制后的帧
    """
    # 【性能优化】避免不必要的复制
    output = frame if inplace else frame.copy()
    
    for result in tracker_results:
        if not result['detected']:
            continue
        
        roi_id = result['roi_id']
        centroid = result['centroid']
        contour = result.get('contour')
        is_struggling = result['is_struggling']
        
        # 颜色：挣扎=红色，不动=绿色
        if result.get('is_swinging', False):
            color = (0, 255, 0) # 绿色
            status_text = 'Immobile (Swing)'
        else:
            color = (0, 0, 255) if is_struggling else (0, 255, 0)
            status_text = 'Struggling' if is_struggling else 'Immobile'
            
        # 绘制轮廓（YOLO全帧分割路径可能没有 contour）
        if contour is not None and isinstance(contour, np.ndarray) and contour.size > 0:
            cv2.drawContours(output, [contour], 0, color, 2)
        
        # 绘制质心
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(output, (cx, cy), 5, color, -1)
        
        roi_name = result.get('roi_name', f"ROI {roi_id}")
        
        # 【性能优化】简化信息显示，减少绘制操作
        # 只显示最重要的 2 行信息
        info_lines = [f"{roi_name} | {status_text}"]
        
        if result['flow'] is not None:
            flow_val = result.get('flow_p90', 0.0)
            ar = result.get('aspect_ratio', 0.0)
            info_lines.append(f"Flow: {flow_val:.1f} | AR: {ar:.2f}")
        
        # 绘制简化的信息（不使用半透明叠加，直接绘制）
        # 用轮廓顶端作为信息框锚点；若无轮廓则用 bbox 顶端
        y_min = cy
        if contour is not None and isinstance(contour, np.ndarray) and contour.size > 0:
            contour_points = contour.reshape(-1, 2)
            y_min = int(np.min(contour_points[:, 1]))
        else:
            bbox = result.get('bbox')
            if bbox is not None and len(bbox) == 4:
                try:
                    y_min = int(bbox[1])
                except Exception:
                    y_min = cy
        
        # 信息框尺寸（更紧凑）
        panel_width = 180
        panel_height = len(info_lines) * 22 + 8
        
        # 位置
        panel_x = max(5, min(cx - panel_width // 2, output.shape[1] - panel_width - 5))
        panel_y = max(10, y_min - panel_height - 30)
        
        # 【性能优化】直接绘制不透明背景（不用 addWeighted）
        cv2.rectangle(output, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(output, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     color, 2)
        
        # 绘制文本
        for i, line in enumerate(info_lines):
            y_pos = panel_y + 18 + i * 22
            cv2.putText(output, line, (panel_x + 5, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 绘制轨迹
        if show_trajectory and 'trajectory' in result:
            trajectory = result['trajectory']
            for i in range(1, len(trajectory)):
                pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                cv2.line(output, pt1, pt2, (200, 200, 200), 1)
    
    return output


# ==================== 便捷函数 ====================

def create_static_background(video_path, num_frames=10, start_frame=0):
    """
    从视频中提取静态背景
    
    参数:
        video_path: 视频路径
        num_frames: 使用多少帧
        start_frame: 起始帧
    
    返回:
        背景图像（中值法）
    """
    cap = cv2.VideoCapture(video_path)
    
    # 跳到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("Failed to read frames from video")
    
    # 中值法生成背景
    bg = np.median(frames, axis=0).astype(np.uint8)
    
    return bg
