from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """한글/유니코드 경로에서도 안전하게 이미지를 읽기 위한 헬퍼."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img


@dataclass
class DefectInfo:
    """단일 결함 정보 (라벨링용 메타데이터)."""

    defect_type: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    params: Dict[str, Any]


def _random_point(h: int, w: int) -> Tuple[int, int]:
    x = np.random.randint(0, w)
    y = np.random.randint(0, h)
    return x, y


_OBJECT_IMAGE_CACHE: Dict[Path, Dict[Path, np.ndarray]] = {}


def load_object_images(object_dir: Path) -> Dict[Path, np.ndarray]:
    """이물(Object) 이미지를 디렉터리(하위 폴더 포함)에서 로드 (캐시 포함)."""
    object_dir = object_dir.resolve()
    if object_dir in _OBJECT_IMAGE_CACHE:
        return _OBJECT_IMAGE_CACHE[object_dir]

    if not object_dir.is_dir():
        _OBJECT_IMAGE_CACHE[object_dir] = {}
        return _OBJECT_IMAGE_CACHE[object_dir]

    images: Dict[Path, np.ndarray] = {}
    exts = {".png", ".jpg", ".jpeg", ".bmp"}

    for p in sorted(object_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        img = imread_unicode(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        images[p] = img

    _OBJECT_IMAGE_CACHE[object_dir] = images
    return images


def add_random_scratches(
    image: np.ndarray,
    count: int = 3,
    min_length: int = 30,
    max_length: int = 150,
    min_thickness: int = 1,
    max_thickness: int = 3,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, List[DefectInfo]]:
    """랜덤 스크래치를 여러 개 생성."""
    h, w = image.shape[:2]
    out = image.copy()
    defects: List[DefectInfo] = []

    for _ in range(count):
        x1, y1 = _random_point(h, w)
        length = np.random.randint(min_length, max_length + 1)
        angle = np.random.uniform(0, 2 * np.pi)
        thickness = np.random.randint(min_thickness, max_thickness + 1)

        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))

        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.line(out, (x1, y1), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)

        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        bbox = (x_min, y_min, max(1, x_max - x_min + thickness), max(1, y_max - y_min + thickness))

        defects.append(
            DefectInfo(
                defect_type="scratch",
                bbox=bbox,
                params={
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "length": length,
                    "angle": float(angle),
                    "thickness": thickness,
                },
            )
        )

    return out, defects


def add_random_spots(
    image: np.ndarray,
    count: int = 5,
    min_radius: int = 3,
    max_radius: int = 15,
    color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.7,
) -> Tuple[np.ndarray, List[DefectInfo]]:
    """랜덤 점(이물/오염) 결함 생성."""
    h, w = image.shape[:2]
    out = image.copy()
    defects: List[DefectInfo] = []

    for _ in range(count):
        x, y = _random_point(h, w)
        radius = np.random.randint(min_radius, max_radius + 1)

        overlay = out.copy()
        cv2.circle(overlay, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

        bbox = (max(0, x - radius), max(0, y - radius), radius * 2, radius * 2)
        defects.append(
            DefectInfo(
                defect_type="spot",
                bbox=bbox,
                params={"x": x, "y": y, "radius": radius, "alpha": alpha},
            )
        )

    return out, defects


def add_object_defects(
    image: np.ndarray,
    object_dir: Path,
    selected_paths: List[Path] | None = None,
    count: int = 5,
    min_scale: float = 0.2,
    max_scale: float = 0.6,
) -> Tuple[np.ndarray, List[DefectInfo]]:
    """Object_list 안의 이물 이미지를 사용해 결함을 합성."""
    h, w = image.shape[:2]
    out = image.copy()

    image_dict = load_object_images(object_dir)
    defects: List[DefectInfo] = []

    if selected_paths:
        candidates = [p for p in selected_paths if p in image_dict]
    else:
        candidates = list(image_dict.keys())

    if not candidates:
        # 이물 이미지가 없으면 기존 랜덤 점 방식으로 폴백
        return add_random_spots(image, count=count)

    for _ in range(count):
        path = candidates[np.random.randint(0, len(candidates))]
        obj = image_dict[path]
        oh, ow = obj.shape[:2]
        if oh == 0 or ow == 0:
            continue

        scale = float(np.random.uniform(min_scale, max_scale))
        new_w = max(1, int(ow * scale))
        new_h = max(1, int(oh * scale))
        if new_w >= w or new_h >= h:
            continue

        obj_resized = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 위치 선택
        x = np.random.randint(0, w - new_w)
        y = np.random.randint(0, h - new_h)

        roi = out[y : y + new_h, x : x + new_w]

        if obj_resized.shape[2] == 4:
            # BGRA → BGR + 알파 추출
            bgr = obj_resized[:, :, :3].astype(np.float32)
            alpha_raw = obj_resized[:, :, 3].astype(np.float32) / 255.0
        else:
            # 알파 채널이 없으면 윤곽 기반 가짜 알파 생성
            gray = cv2.cvtColor(obj_resized, cv2.COLOR_BGR2GRAY)
            alpha_raw = gray.astype(np.float32) / 255.0

        # 알파 마스크 소프트닝으로 경계 자연스럽게
        alpha = cv2.GaussianBlur(alpha_raw, (0, 0), sigmaX=1.5)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_3 = alpha[..., None]

        bgr = obj_resized[:, :, :3].astype(np.float32)
        roi_f = roi.astype(np.float32)

        # 결함 색을 배경에 맞게 보정 (밝기/색감 간단 매칭) - 주로 테두리 부근에만 적용
        mask = alpha > 0.1
        edge_mask = (alpha > 0.1) & (alpha < 0.9)
        stats_mask = edge_mask if np.any(edge_mask) else mask
        if np.any(stats_mask):
            obj_mean = bgr[stats_mask].mean(axis=0)
            roi_mean = roi_f[stats_mask].mean(axis=0)

            # 0으로 나누는 것 방지
            obj_mean = np.clip(obj_mean, 1.0, None)
            gain = roi_mean / obj_mean
            # 과도한 색 변화 방지
            gain = np.clip(gain, 0.8, 1.2)

            # 테두리에서만 강하게, 내부는 거의 영향 없도록 가중치 부여
            edge_weight = np.zeros_like(alpha, dtype=np.float32)
            edge_weight[edge_mask] = 1.0
            bgr = bgr * (1.0 + edge_weight[..., None] * (gain - 1.0))

        # 전역 투명도 조정 (너무 옅어지지 않게 범위 축소)
        global_alpha = float(np.random.uniform(0.8, 1.0))
        alpha_3 = alpha_3 * global_alpha

        blended = bgr * alpha_3 + roi_f * (1.0 - alpha_3)
        out[y : y + new_h, x : x + new_w] = np.clip(blended, 0, 255).astype(np.uint8)

        bbox = (x, y, new_w, new_h)
        defects.append(
            DefectInfo(
                defect_type="foreign_object",
                bbox=bbox,
                params={
                    "scale": scale,
                    "object_size": (ow, oh),
                    "placed_size": (new_w, new_h),
                    "source_path": str(path),
                },
            )
        )

    return out, defects


def apply_object_at(
    image: np.ndarray,
    object_dir: Path,
    object_path: Path,
    x: int,
    y: int,
    w: int,
    h: int,
    angle: float = 0.0,
    use_correction: bool = True,
) -> Tuple[np.ndarray, List[DefectInfo]]:
    """단일 이물을 사용자가 지정한 위치/크기로 합성.

    x, y 좌표는 기본적으로 "중심"으로 간주하고, w/h가 0 이하이면
    이미지 크기의 일정 비율로 자동 스케일링한다.
    """
    h_img, w_img = image.shape[:2]

    out = image.copy()
    image_dict = load_object_images(object_dir)
    # 캐시에 없을 수도 있으니 resolve 후 다시 시도
    obj = image_dict.get(object_path.resolve())
    if obj is None:
        obj = imread_unicode(object_path, cv2.IMREAD_UNCHANGED)
        if obj is None:
            return out, []

    oh, ow = obj.shape[:2]
    if oh == 0 or ow == 0:
        return out, []

    # 크기 자동 결정 (w/h <= 0 이면)
    if w <= 0 or h <= 0:
        # 이미지 폭의 15% 정도를 기본 크기로 사용
        base_w = int(w_img * 0.15)
        base_w = max(8, min(base_w, ow))
        scale = base_w / float(ow)
        w = int(ow * scale)
        h = int(oh * scale)

    # 중심 좌표(x, y)를 좌상단으로 변환
    x = int(x - w / 2)
    y = int(y - h / 2)

    # 이미지 경계를 벗어나지 않도록 클램핑
    x = max(0, min(w_img - 1, x))
    y = max(0, min(h_img - 1, y))
    w = max(1, min(w_img - x, w))
    h = max(1, min(h_img - y, h))

    # 지정된 영역 크기에 맞게 리사이즈
    obj_resized = cv2.resize(obj, (w, h), interpolation=cv2.INTER_AREA)

    # 회전 적용 (필요 시) - 빈 공간은 투명/검정으로 채움
    if abs(angle) > 1e-2:
        center_rot = (w / 2.0, h / 2.0)
        # Qt의 화면 회전 방향과 OpenCV의 회전 방향이 반대라 부호를 반전
        rot_mat = cv2.getRotationMatrix2D(center_rot, -angle, 1.0)
        if obj_resized.shape[2] == 4:
            border_value = (0, 0, 0, 0)
        else:
            border_value = (0, 0, 0)
        obj_resized = cv2.warpAffine(
            obj_resized,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    roi = out[y : y + h, x : x + w]

    if obj_resized.shape[2] == 4:
        # BGRA → BGR + 알파 추출
        bgr = obj_resized[:, :, :3].astype(np.float32)
        alpha_raw = obj_resized[:, :, 3].astype(np.float32) / 255.0
    else:
        # 알파 채널이 없으면 윤곽 기반 가짜 알파 생성
        gray = cv2.cvtColor(obj_resized, cv2.COLOR_BGR2GRAY)
        alpha_raw = gray.astype(np.float32) / 255.0

    # 알파 마스크 소프트닝으로 경계 자연스럽게 (보정 사용 시에만)
    if use_correction:
        alpha = cv2.GaussianBlur(alpha_raw, (0, 0), sigmaX=1.5)
    else:
        alpha = alpha_raw
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha_3 = alpha[..., None]

    bgr = obj_resized[:, :, :3].astype(np.float32)
    roi_f = roi.astype(np.float32)

    # 결함 색을 배경에 맞게 보정 (밝기/색감 간단 매칭) - 주로 테두리 부근에만 적용
    if use_correction:
        mask = alpha > 0.1
        edge_mask = (alpha > 0.1) & (alpha < 0.9)
        stats_mask = edge_mask if np.any(edge_mask) else mask
        if np.any(stats_mask):
            obj_mean = bgr[stats_mask].mean(axis=0)
            roi_mean = roi_f[stats_mask].mean(axis=0)

            obj_mean = np.clip(obj_mean, 1.0, None)
            gain = roi_mean / obj_mean
            gain = np.clip(gain, 0.8, 1.2)

            edge_weight = np.zeros_like(alpha, dtype=np.float32)
            edge_weight[edge_mask] = 1.0
            bgr = bgr * (1.0 + edge_weight[..., None] * (gain - 1.0))

        global_alpha = float(np.random.uniform(0.8, 1.0))
        alpha_3 = alpha_3 * global_alpha
    else:
        # 보정 OFF: 색/밝기 보정 없이 알파만 그대로 사용
        global_alpha = 1.0
        alpha_3 = alpha_3 * global_alpha

    blended = bgr * alpha_3 + roi_f * (1.0 - alpha_3)
    out[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)

    defect = DefectInfo(
        defect_type="foreign_object",
        bbox=(x, y, w, h),
        params={
            "scale": float(w) / float(ow),
            "object_size": (ow, oh),
            "placed_size": (w, h),
            "source_path": str(object_path),
            "angle": float(angle),
        },
    )

    return out, [defect]


def add_noise(
    image: np.ndarray,
    sigma: float = 10.0,
) -> Tuple[np.ndarray, List[DefectInfo]]:
    """이미지 전체에 가우시안 노이즈를 추가 (라벨 1개로 취급)."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    h, w = image.shape[:2]
    defect = DefectInfo(
        defect_type="noise",
        bbox=(0, 0, w, h),
        params={"sigma": sigma},
    )
    return noisy, [defect]


def apply_defects(
    image: np.ndarray,
    use_scratch: bool = True,
    use_spot: bool = True,
    use_noise: bool = False,
    object_dir: Path | None = None,
) -> Tuple[np.ndarray, List[DefectInfo]]:
    """여러 종류의 결함을 한 번에 적용."""
    out = image.copy()
    all_defects: List[DefectInfo] = []

    if use_scratch:
        out, defects = add_random_scratches(out)
        all_defects.extend(defects)

    if use_spot:
        if object_dir is not None:
            out, defects = add_object_defects(out, object_dir=object_dir, selected_paths=None)
            all_defects.extend(defects)
        else:
            out, defects = add_random_spots(out)
            all_defects.extend(defects)

    if use_noise:
        out, defects = add_noise(out)
        all_defects.extend(defects)

    return out, all_defects



