from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import random
from math import atan2, degrees, cos, sin, radians
from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal, QMimeData
from PySide6.QtGui import QAction, QDrag, QIcon, QImage, QPainter, QPen, QPixmap, QColor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QMessageBox,
    QCheckBox,
)

from defects import DefectInfo, apply_object_at


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """한글/유니코드 경로에서도 안전하게 이미지를 읽기 위한 헬퍼."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """한글/유니코드 경로에서도 안전하게 이미지를 저장하기 위한 헬퍼."""
    ext = os.path.splitext(str(path))[1]
    if not ext:
        ext = ".png"
    success, buf = cv2.imencode(ext, image)
    if not success:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


class ObjectListWidget(QListWidget):
    """썸네일을 이미지 뷰어로 드래그하기 위한 전용 리스트."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setDragEnabled(True)

    def startDrag(self, supportedActions):  # noqa: N802
        item = self.currentItem()
        if item is None:
            return
        path_str = item.data(Qt.UserRole)
        if not path_str:
            return

        mime = QMimeData()
        mime.setText(path_str)

        drag = QDrag(self)
        drag.setMimeData(mime)

        icon = item.icon()
        if not icon.isNull():
            drag.setPixmap(icon.pixmap(80, 80))

        drag.exec(Qt.CopyAction)


class ImageListWidget(QListWidget):
    """현재 선택을 유지하기 위한 이미지 리스트 (빈 영역 클릭 시 선택 해제 금지)."""

    def mousePressEvent(self, event):  # noqa: N802
        # PySide6에서는 event.pos()가 deprecated 이므로 position()을 사용
        pos = event.position().toPoint()
        item = self.itemAt(pos)
        if item is None:
            # 빈 영역 클릭 시 선택 해제하지 않음
            event.ignore()
            return
        super().mousePressEvent(event)


def cvimg_to_qpixmap(img: np.ndarray) -> QPixmap:
    """OpenCV BGR 이미지를 QPixmap으로 변환."""
    if img is None:
        return QPixmap()

    if len(img.shape) == 2:
        h, w = img.shape
        bytes_per_line = w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

    return QPixmap.fromImage(qimg)


def make_icon_pixmap(path: str, size: QSize = QSize(80, 80)) -> QPixmap:
    """이물 썸네일용 아이콘 픽셀을 고정 크기로 생성 (텍스트 정렬을 위해)."""
    pix = QPixmap(path)
    if pix.isNull():
        return QPixmap()
    scaled = pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    canvas = QPixmap(size)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    x = (size.width() - scaled.width()) // 2
    y = (size.height() - scaled.height()) // 2
    painter.drawPixmap(x, y, scaled)
    painter.end()
    return canvas


class ImageViewer(QLabel):
    """이미지 표시 + 드래그로 영역 선택 기능을 가진 뷰어."""

    rectFinished = Signal(int, int, int, int)  # x, y, w, h (이미지 좌표계)
    objectDropped = Signal(str, int, int)  # path, x, y (이미지 좌표계 중심)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("이미지를 불러오세요")
        self.setStyleSheet("color: #888; font-size: 16px;")
        self._pixmap: QPixmap | None = None
        self._scaled: QPixmap | None = None
        self._selection_enabled: bool = False
        self.setAcceptDrops(True)
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        # 오버레이(임시 이물 미리보기)
        self._overlay_pixmap: QPixmap | None = None
        self._overlay_rect_img: QRect | None = None  # 이미지 좌표계
        self._overlay_dragging: bool = False
        self._overlay_drag_offset: QPoint | None = None  # 이미지 좌표계 기준
        self._overlay_resizing: bool = False
        self._overlay_resize_anchor: QPoint | None = None  # 이미지 좌표계 기준 (고정 코너)
        self._overlay_angle: float = 0.0
        self._overlay_rotating: bool = False
        self._overlay_rotate_start_angle: float = 0.0
        self._overlay_rotate_initial_angle: float = 0.0
        # 회전 아이콘 이미지 (rotation.png)
        self._rot_icon: QPixmap | None = QPixmap("rotation.png")
        if self._rot_icon is not None and self._rot_icon.isNull():
            self._rot_icon = None

    def set_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._update_scaled_pixmap()

    def enable_selection(self, enabled: bool):
        self._selection_enabled = enabled
        self._drag_start = None
        self._drag_current = None
        self.update()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._pixmap is None or self.width() <= 0 or self.height() <= 0:
            self._scaled = None
            self.setPixmap(QPixmap())
            return
        self._scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(self._scaled)

    def _image_geometry(self) -> tuple[float, float, float] | None:
        """라벨 안에서 실제 이미지가 차지하는 위치/스케일 반환: (offset_x, offset_y, scale)."""
        if self._pixmap is None or self._scaled is None:
            return None
        w_label = self.width()
        h_label = self.height()
        w_scaled = self._scaled.width()
        h_scaled = self._scaled.height()
        offset_x = (w_label - w_scaled) / 2.0
        offset_y = (h_label - h_scaled) / 2.0
        scale = w_scaled / self._pixmap.width()
        return offset_x, offset_y, scale

    def _widget_to_image_point(self, pt: QPoint) -> QPoint | None:
        geo = self._image_geometry()
        if geo is None:
            return None
        offset_x, offset_y, scale = geo
        x = (pt.x() - offset_x) / scale
        y = (pt.y() - offset_y) / scale
        return QPoint(int(x), int(y))

    def set_overlay(self, pixmap: QPixmap | None, rect_img: QRect | None):
        """임시 이물 오버레이 설정 (이미지 좌표계 기준)."""
        self._overlay_pixmap = pixmap
        self._overlay_rect_img = QRect(rect_img) if rect_img is not None else None
        self._overlay_dragging = False
        self._overlay_drag_offset = None
        self._overlay_resizing = False
        self._overlay_resize_anchor = None
        self._overlay_angle = 0.0
        self._overlay_rotating = False
        self.update()

    def get_overlay_rect(self) -> QRect | None:
        """현재 오버레이 사각형(이미지 좌표계) 반환."""
        if self._overlay_rect_img is None:
            return None
        return QRect(self._overlay_rect_img)

    def get_overlay_angle(self) -> float:
        return float(self._overlay_angle)

    def _hit_test_overlay_handle(self, img_pt: QPoint) -> str | None:
        """오버레이 코너 핸들 히트 테스트 (이미지 좌표계)."""
        if self._overlay_rect_img is None:
            return None
        r = self._overlay_rect_img
        # 회전된 상태에서도 시각적으로 보이는 코너 근처를 클릭하면
        # 크기 조절이 되도록, 클릭 지점을 -angle 만큼 역회전시켜
        # 원래 축 기준 좌표계로 변환한 뒤 히트 테스트를 수행한다.
        center = r.center()
        angle_rad = radians(-self._overlay_angle)
        dx_click = img_pt.x() - center.x()
        dy_click = img_pt.y() - center.y()
        ux = dx_click * cos(angle_rad) - dy_click * sin(angle_rad)
        uy = dx_click * sin(angle_rad) + dy_click * cos(angle_rad)
        unrotated_pt = QPoint(int(center.x() + ux), int(center.y() + uy))

        handles = {
            "tl": r.topLeft(),
            "tr": r.topRight(),
            "bl": r.bottomLeft(),
            "br": r.bottomRight(),
        }
        # 이미지 좌표계에서 허용 거리는 대략 10픽셀
        thresh2 = 10 * 10
        for name, pt in handles.items():
            dx = unrotated_pt.x() - pt.x()
            dy = unrotated_pt.y() - pt.y()
            if dx * dx + dy * dy <= thresh2:
                return name
        return None

    def _hit_test_rotation_handle(self, img_pt: QPoint) -> bool:
        """오버레이 회전 핸들 히트 테스트 (이미지 좌표계)."""
        if self._overlay_rect_img is None:
            return False
        r = self._overlay_rect_img
        center = r.center()
        # 로컬 좌표계에서 핸들은 상단 중앙에서 약간 위쪽(handle_offset) 위치
        handle_offset = r.height() / 2.0 + 18.0
        angle_rad = radians(self._overlay_angle)
        # (0, -handle_offset)을 회전시켜 이미지 좌표계로 변환
        dx_local = 0.0
        dy_local = -handle_offset
        dx = dx_local * cos(angle_rad) - dy_local * sin(angle_rad)
        dy = dx_local * sin(angle_rad) + dy_local * cos(angle_rad)
        handle_pt = QPoint(int(center.x() + dx), int(center.y() + dy))
        dx = img_pt.x() - handle_pt.x()
        dy = img_pt.y() - handle_pt.y()
        # 클릭 범위를 넓히기 위해 반지름을 키운다 (기존 12px → 20px)
        return dx * dx + dy * dy <= 20 * 20

    def _widget_to_image_rect(self, rect: QRect) -> QRect | None:
        geo = self._image_geometry()
        if geo is None:
            return None
        offset_x, offset_y, scale = geo

        x1 = (rect.left() - offset_x) / scale
        y1 = (rect.top() - offset_y) / scale
        x2 = (rect.right() - offset_x) / scale
        y2 = (rect.bottom() - offset_y) / scale

        ix1 = int(np.floor(min(x1, x2)))
        iy1 = int(np.floor(min(y1, y2)))
        ix2 = int(np.ceil(max(x1, x2)))
        iy2 = int(np.ceil(max(y1, y2)))

        if ix2 <= ix1 or iy2 <= iy1:
            return None

        return QRect(ix1, iy1, ix2 - ix1, iy2 - iy1)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            pos = event.position().toPoint()
            img_pt = self._widget_to_image_point(pos)
            # 오버레이가 있고, 그 안을 클릭하면 이동 모드
            if self._overlay_pixmap is not None and self._overlay_rect_img is not None and img_pt is not None:
                # 회전 핸들 클릭 여부 확인
                if self._hit_test_rotation_handle(img_pt):
                    center = self._overlay_rect_img.center()
                    v = img_pt - center
                    self._overlay_rotate_start_angle = degrees(atan2(v.y(), v.x()))
                    self._overlay_rotate_initial_angle = self._overlay_angle
                    self._overlay_rotating = True
                    event.accept()
                    return
                # 코너 핸들 클릭 여부 확인 (리사이즈 모드)
                handle = self._hit_test_overlay_handle(img_pt)
                if handle is not None:
                    self._overlay_resizing = True
                    # 고정 코너(반대쪽) 설정
                    r = self._overlay_rect_img
                    if handle == "tl":
                        self._overlay_resize_anchor = r.bottomRight()
                    elif handle == "tr":
                        self._overlay_resize_anchor = r.bottomLeft()
                    elif handle == "bl":
                        self._overlay_resize_anchor = r.topRight()
                    else:  # "br"
                        self._overlay_resize_anchor = r.topLeft()
                    event.accept()
                    return
                # 핸들이 아니라 내부 클릭이면 이동 모드
                if self._overlay_rect_img.contains(img_pt):
                    self._overlay_dragging = True
                    top_left = self._overlay_rect_img.topLeft()
                    self._overlay_drag_offset = img_pt - top_left
                    event.accept()
                    return
            # 영역 선택 모드
            if self._selection_enabled:
                self._drag_start = pos
                self._drag_current = self._drag_start
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._overlay_rotating and self._overlay_rect_img is not None:
            img_pt = self._widget_to_image_point(event.position().toPoint())
            if img_pt is not None:
                center = self._overlay_rect_img.center()
                v = img_pt - center
                cur_angle = degrees(atan2(v.y(), v.x()))
                delta = cur_angle - self._overlay_rotate_start_angle
                self._overlay_angle = self._overlay_rotate_initial_angle + delta
                self.update()
            event.accept()
            return
        if self._overlay_resizing and self._overlay_rect_img is not None:
            img_pt = self._widget_to_image_point(event.position().toPoint())
            if img_pt is not None and self._overlay_resize_anchor is not None and self._pixmap is not None:
                # 클릭 지점을 역회전시켜 축 기준 좌표계에서 리사이즈
                center = self._overlay_rect_img.center()
                angle_rad = radians(-self._overlay_angle)
                dx = img_pt.x() - center.x()
                dy = img_pt.y() - center.y()
                ux = dx * cos(angle_rad) - dy * sin(angle_rad)
                uy = dx * sin(angle_rad) + dy * cos(angle_rad)
                unrot_pt = QPoint(int(center.x() + ux), int(center.y() + uy))

                # 고정 코너(self._overlay_resize_anchor)와 역회전된 현재 포인트로
                # 새 사각형을 만들고, normalized()로 항상 왼쪽/위가 작게 정렬
                new_rect = QRect(self._overlay_resize_anchor, unrot_pt).normalized()

                # 최소 크기 제한
                if new_rect.width() < 8 or new_rect.height() < 8:
                    event.accept()
                    return

                # 이미지 경계 내로 제한
                max_w = self._pixmap.width()
                max_h = self._pixmap.height()
                x = max(0, min(max_w - new_rect.width(), new_rect.x()))
                y = max(0, min(max_h - new_rect.height(), new_rect.y()))
                new_rect.moveTo(x, y)

                self._overlay_rect_img = new_rect
                self.update()
            event.accept()
            return
        if self._overlay_dragging and self._overlay_rect_img is not None:
            img_pt = self._widget_to_image_point(event.position().toPoint())
            if img_pt is not None and self._overlay_drag_offset is not None and self._pixmap is not None:
                new_top_left = img_pt - self._overlay_drag_offset
                # 이미지 경계 내로 제한
                x = max(0, min(self._pixmap.width() - self._overlay_rect_img.width(), new_top_left.x()))
                y = max(0, min(self._pixmap.height() - self._overlay_rect_img.height(), new_top_left.y()))
                self._overlay_rect_img.moveTo(x, y)
                self.update()
            event.accept()
            return
        if self._selection_enabled and self._drag_start is not None:
            self._drag_current = event.position().toPoint()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            if self._overlay_rotating:
                self._overlay_rotating = False
                event.accept()
                return
            if self._overlay_resizing:
                self._overlay_resizing = False
                self._overlay_resize_anchor = None
                event.accept()
                return
            if self._overlay_dragging:
                self._overlay_dragging = False
                self._overlay_drag_offset = None
                event.accept()
                return
            if self._selection_enabled and self._drag_start is not None:
                self._drag_current = event.position().toPoint()
                rect_widget = QRect(self._drag_start, self._drag_current)
                rect_image = self._widget_to_image_rect(rect_widget)
                self._drag_start = None
                self._drag_current = None
                self.update()

                if rect_image is not None and rect_image.width() > 5 and rect_image.height() > 5:
                    self.rectFinished.emit(
                        rect_image.x(),
                        rect_image.y(),
                        rect_image.width(),
                        rect_image.height(),
                    )
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if self._selection_enabled and self._drag_start is not None and self._drag_current is not None:
            painter = QPainter(self)
            pen = QPen(QColor(0, 200, 255), 2, Qt.DashLine)
            painter.setPen(pen)
            brush_color = QColor(0, 200, 255, 50)
            painter.setBrush(brush_color)
            rect = QRect(self._drag_start, self._drag_current)
            painter.drawRect(rect.normalized())
            painter.end()

        # 이물 오버레이 미리보기
        if self._overlay_pixmap is not None and self._overlay_rect_img is not None and self._pixmap is not None:
            geo = self._image_geometry()
            if geo is not None:
                offset_x, offset_y, scale = geo
                painter = QPainter(self)
                img_rect = self._overlay_rect_img
                x = offset_x + img_rect.x() * scale
                y = offset_y + img_rect.y() * scale
                w = img_rect.width() * scale
                h = img_rect.height() * scale
                target = QRect(int(x), int(y), int(w), int(h))
                overlay_scaled = self._overlay_pixmap.scaled(
                    target.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                # 회전된 오버레이 이미지 + 초록색 박스/코너 핸들
                painter.save()
                center = target.center()
                painter.translate(center)
                painter.rotate(self._overlay_angle)
                draw_rect = QRect(
                    -target.width() // 2,
                    -target.height() // 2,
                    target.width(),
                    target.height(),
                )
                # 실제 이물 이미지
                painter.drawPixmap(draw_rect, overlay_scaled)
                # 이물에 맞춰 회전된 초록색 박스
                pen = QPen(QColor(0, 255, 0), 1, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(draw_rect)
                # 코너 핸들 (박스와 같이 회전)
                handle_size = 6
                half = handle_size // 2
                corners = [
                    draw_rect.topLeft(),
                    draw_rect.topRight(),
                    draw_rect.bottomLeft(),
                    draw_rect.bottomRight(),
                ]
                painter.setBrush(QColor(0, 255, 0))
                for c in corners:
                    painter.drawRect(c.x() - half, c.y() - half, handle_size, handle_size)

                # 회전 핸들 (rotation.png 아이콘 사용) - 상단 중앙에서 박스와 함께 회전
                handle_y_offset = 18
                center_pt = QPoint(
                    0,
                    -draw_rect.height() // 2 - handle_y_offset,
                )
                if self._rot_icon is not None:
                    icon_size = 20
                    icon_rect = QRect(
                        center_pt.x() - icon_size // 2,
                        center_pt.y() - icon_size // 2,
                        icon_size,
                        icon_size,
                    )
                    painter.drawPixmap(
                        icon_rect,
                        self._rot_icon.scaled(
                            icon_rect.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation,
                        ),
                    )
                else:
                    # fallback: 작은 원으로 표시
                    painter.setBrush(QColor(0, 200, 255))
                    painter.drawEllipse(center_pt, 6, 6)

                painter.restore()
                painter.end()

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):  # noqa: N802
        if not event.mimeData().hasText() or self._pixmap is None:
            super().dropEvent(event)
            return

        geo = self._image_geometry()
        if geo is None:
            return
        offset_x, offset_y, scale = geo

        pos = event.position().toPoint()
        ix = int((pos.x() - offset_x) / scale)
        iy = int((pos.y() - offset_y) / scale)

        self.objectDropped.emit(event.mimeData().text(), ix, iy)
        event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Generation")
        self.resize(1400, 800)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        central_layout.addWidget(splitter)

        # 왼쪽 툴바 영역
        self.toolbar_widget = QWidget()
        toolbar_layout = QVBoxLayout(self.toolbar_widget)
        toolbar_layout.setContentsMargins(4, 8, 4, 8)
        toolbar_layout.setSpacing(8)

        # 이물 카테고리 버튼들 (고무/비닐/머리카락 등)
        self.category_buttons: dict[str, QPushButton] = {}

        # 중앙 이미지 뷰어
        self.viewer = ImageViewer()

        # 우측 패널: 이미지 폴더/파일 리스트 + 이물 썸네일 + 조작 버튼들
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        # 이미지 폴더 선택
        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(8)

        self.btn_select_folder = QPushButton("이미지 폴더 선택")
        self.btn_select_folder.clicked.connect(self.select_image_folder)
        folder_layout.addWidget(self.btn_select_folder)

        self.current_folder_label = QLabel("")
        self.current_folder_label.setStyleSheet("color: #aaa;")
        folder_layout.addWidget(self.current_folder_label, 1)

        right_layout.addWidget(folder_row)

        # 폴더 내 이미지 파일 리스트
        self.image_list = ImageListWidget()
        self.image_list.setSelectionMode(QListWidget.SingleSelection)
        self.image_list.itemClicked.connect(self.on_image_selected)
        right_layout.addWidget(self.image_list, 1)

        # 이물 조작 버튼들 (원본/보정 선택 + 적용 / 되돌리기 / 초기화 / 저장)
        buttons_row = QWidget()
        btn_layout = QHBoxLayout(buttons_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)

        self.chk_use_correction = QCheckBox("보정")
        self.chk_use_correction.setChecked(True)
        btn_layout.addWidget(self.chk_use_correction)

        self.btn_apply = QPushButton("적용")
        self.btn_apply.clicked.connect(self.apply_selected_object)
        btn_layout.addWidget(self.btn_apply)

        self.btn_cancel_overlay = QPushButton("되돌리기")
        self.btn_cancel_overlay.clicked.connect(self.cancel_overlay)
        btn_layout.addWidget(self.btn_cancel_overlay)

        self.btn_reset_all = QPushButton("초기화")
        self.btn_reset_all.clicked.connect(self.reset_image)
        btn_layout.addWidget(self.btn_reset_all)

        self.btn_save = QPushButton("저장")
        self.btn_save.clicked.connect(self.save_result)
        btn_layout.addWidget(self.btn_save)

        right_layout.addWidget(buttons_row)

        self.thumb_list = ObjectListWidget()
        self.thumb_list.setViewMode(QListWidget.IconMode)
        # 이물 아이콘 크기는 기존대로 유지
        self.thumb_list.setIconSize(QSize(100, 100))
        self.thumb_list.setResizeMode(QListWidget.Adjust)
        self.thumb_list.setSpacing(8)
        self.thumb_list.setSelectionMode(QListWidget.SingleSelection)
        right_layout.addWidget(self.thumb_list, 2)

        splitter.addWidget(self.toolbar_widget)
        splitter.addWidget(self.viewer)
        splitter.addWidget(self.right_panel)

        # 좌/중/우 비율 설정 (오른쪽 전체 박스를 더 좁게)
        splitter.setSizes([120, 1000, 230])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self.setCentralWidget(central)

        # 상태
        self.original_image: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.defects: List[DefectInfo] = []
        self.current_image_path: Path | None = None
        # 이물(Object) 이미지 디렉터리 (필요 시 경로 수정)
        self.object_dir: Path = Path(r"Object_list")
        self.object_categories: dict[str, list[Path]] = {}
        self.current_category: str | None = None
        self.pending_object_path: Path | None = None
        self.overlay_object_path: Path | None = None
        self.dirty: bool = False
        self.current_folder_path: Path | None = None

        # 카테고리 스캔 후 왼쪽 버튼 + 오른쪽 썸네일 초기화
        self._scan_object_categories()
        self._build_category_buttons(toolbar_layout)
        toolbar_layout.addStretch(1)

        # 드래그 선택 / 드롭 완료 시 콜백 연결
        self.viewer.rectFinished.connect(self._on_rect_finished)
        self.viewer.objectDropped.connect(self._on_object_dropped)

        # 전체 다크 테마 느낌
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #202124;
            }
            QWidget {
                background-color: #202124;
                color: #f5f5f5;
            }
            QPushButton {
                background-color: #303134;
                border: 1px solid #3c4043;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #3c4043;
            }
            QPushButton:checked {
                background-color: #2563eb;
                border-color: #3b82f6;
                color: #ffffff;
            }
            QListWidget {
                background-color: #18181b;
                border: 1px solid #3c4043;
            }
            QListWidget::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
            QListWidget::item:selected:!active {
                background-color: #1d4ed8;
                color: #ffffff;
            }
            """
        )

    # 이미지 로딩 및 표시
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "이미지 선택",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return

        self._load_image_from_path(file_path)

    def _load_image_from_path(self, file_path: str | Path):
        if not file_path:
            return
        img = imread_unicode(file_path, cv2.IMREAD_COLOR)
        if img is None:
            return

        self.original_image = img
        self.current_image = img.copy()
        self.current_image_path = Path(file_path)
        self.defects.clear()
        self.dirty = False
        self.viewer.set_overlay(None, None)
        self.overlay_object_path = None
        self.viewer.set_image(cvimg_to_qpixmap(self.current_image))
        self._sync_image_list_selection()

    def _scan_object_categories(self):
        """Object_list 폴더의 이물 카테고리를 스캔."""
        self.object_categories.clear()

        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        root = self.object_dir
        if not root.is_dir():
            return

        # 하위 폴더를 카테고리로 사용
        for child in sorted(root.iterdir()):
            if child.is_dir():
                paths: list[Path] = [
                    p for p in child.rglob("*") if p.is_file() and p.suffix.lower() in exts
                ]
                if paths:
                    self.object_categories[child.name] = paths
            else:
                if child.suffix.lower() in exts:
                    self.object_categories.setdefault("기타", []).append(child)

    def _build_category_buttons(self, layout: QVBoxLayout):
        """왼쪽에 이물 카테고리 버튼 생성."""
        for name in sorted(self.object_categories.keys()):
            btn = QPushButton(name)
            btn.setCheckable(True)
            # clicked(bool) 시그널과 호환되도록 래퍼 사용
            btn.clicked.connect(self._make_category_handler(name))
            layout.addWidget(btn)
            self.category_buttons[name] = btn

        # 첫 번째 카테고리를 기본 선택
        if self.category_buttons:
            first = next(iter(sorted(self.category_buttons.keys())))
            self.on_category_clicked(first)

    def _build_auto_category_panel(self):
        """자동 모드용 이물 카테고리 체크/확률 UI 구성."""
        # 기존 위젯 비우기
        while self.auto_category_layout.count():
            item = self.auto_category_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.auto_category_checks.clear()
        self.auto_category_probs.clear()

        for name in sorted(self.object_categories.keys()):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            chk = QPushButton(name)
            chk.setCheckable(True)
            chk.setChecked(True)
            row_layout.addWidget(chk)

            lbl_prob = QLabel("확률(%)")
            row_layout.addWidget(lbl_prob)

            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(100)
            row_layout.addWidget(spin)

            row_layout.addStretch(1)

            self.auto_category_layout.addWidget(row)
            self.auto_category_checks[name] = chk
            self.auto_category_probs[name] = spin

        self.auto_category_layout.addStretch(1)

    def _make_category_handler(self, category: str):
        """카테고리 버튼용 슬롯 생성 (clicked(bool)과 시그니처 맞추기)."""

        def handler(checked: bool = False):  # noqa: ARG001 - Qt 시그널용
            self.on_category_clicked(category)

        return handler

    def on_category_clicked(self, category: str):
        """왼쪽 카테고리 버튼 클릭 시 썸네일 리스트 업데이트."""
        self.current_category = category
        for name, btn in self.category_buttons.items():
            btn.setChecked(name == category)
        self._populate_thumbnails()

    def _populate_thumbnails(self):
        self.thumb_list.clear()
        if not self.current_category:
            return
        paths = self.object_categories.get(self.current_category, [])
        for idx, p in enumerate(paths, start=1):
            item = QListWidgetItem()
            # 파일명 대신 "n번 이미지"로 표시
            item.setText(f"{idx}번 이미지")
            item.setData(Qt.UserRole, str(p))
            item.setIcon(QIcon(make_icon_pixmap(str(p), self.thumb_list.iconSize())))
            self.thumb_list.addItem(item)

    # 우측 이미지 폴더/파일 리스트 관련
    def select_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택", "")
        if not folder:
            return
        folder_path = Path(folder)
        self.current_folder_label.setText(str(folder_path))
        self.current_folder_path = folder_path

        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.image_list.clear()
        for p in sorted(folder_path.iterdir()):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            item = QListWidgetItem(p.name)
            item.setData(Qt.UserRole, str(p))
            self.image_list.addItem(item)
        self._sync_image_list_selection()

    def on_image_selected(self, item: QListWidgetItem):
        path_str = item.data(Qt.UserRole)
        if not path_str:
            return
        # 변경 내용이 있으면 저장 여부 확인
        if self.dirty:
            res = QMessageBox.question(
                self,
                "변경 내용 저장",
                "현재 이미지의 변경 내용을 저장하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if res == QMessageBox.Cancel:
                self._sync_image_list_selection()
                return
            if res == QMessageBox.Yes:
                if not self.save_result(ask_if_no_base=False):
                    self._sync_image_list_selection()
                    return
        self._load_image_from_path(path_str)

    # 자동 탭용 이미지 폴더/파일 리스트
    def select_image_folder_auto(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택 (자동)", "")
        if not folder:
            return
        folder_path = Path(folder)
        self.current_folder_label_auto.setText(str(folder_path))
        self.current_folder_path_auto = folder_path

        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.image_list_auto.clear()
        for p in sorted(folder_path.iterdir()):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            item = QListWidgetItem(p.name)
            item.setData(Qt.UserRole, str(p))
            self.image_list_auto.addItem(item)

    def on_auto_image_selected(self, item: QListWidgetItem):
        path_str = item.data(Qt.UserRole)
        if not path_str:
            return
        img = imread_unicode(path_str, cv2.IMREAD_COLOR)
        if img is None:
            return
        self.viewer_auto.set_image(cvimg_to_qpixmap(img))

    def _sync_image_list_selection(self):
        """현재 표시 중인 이미지를 우측 파일 리스트에서 강조."""
        if self.current_image_path is None:
            self.image_list.clearSelection()
            return
        cur = str(self.current_image_path)
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.UserRole) == cur:
                self.image_list.setCurrentItem(item)
                self.image_list.scrollToItem(item)
                return
        self.image_list.clearSelection()

    def _add_image_to_list(self, path: Path, select: bool = False):
        """저장된 이미지를 현재 폴더 리스트에 추가."""
        if self.current_folder_path is None or path.parent != self.current_folder_path:
            return
        path_str = str(path)
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.UserRole) == path_str:
                if select:
                    self.image_list.setCurrentItem(item)
                    self.image_list.scrollToItem(item)
                return
        item = QListWidgetItem(path.name)
        item.setData(Qt.UserRole, path_str)
        self.image_list.addItem(item)
        if select:
            self.image_list.setCurrentItem(item)
            self.image_list.scrollToItem(item)

    def apply_selected_object(self):
        """현재 오버레이 상태를 실제 이미지에 적용."""
        if self.current_image is None:
            return
        if self.overlay_object_path is None:
            return
        rect = self.viewer.get_overlay_rect()
        if rect is None:
            return

        center = rect.center()
        angle = self.viewer.get_overlay_angle()
        use_corr = self.chk_use_correction.isChecked()
        out, defects = apply_object_at(
            self.current_image,
            self.object_dir,
            self.overlay_object_path,
            center.x(),
            center.y(),
            rect.width(),
            rect.height(),
            angle,
            use_corr,
        )
        self.current_image = out
        self.defects.extend(defects)
        self.viewer.set_image(cvimg_to_qpixmap(self.current_image))
        # 오버레이 초기화
        self.overlay_object_path = None
        self.viewer.set_overlay(None, None)
        self.dirty = True

    def cancel_overlay(self):
        """현재 미리보기 오버레이만 제거 (이미지에는 영향 없음)."""
        self.overlay_object_path = None
        self.viewer.set_overlay(None, None)

    def _on_rect_finished(self, x: int, y: int, w: int, h: int):
        """이미지 위에서 드래그 영역이 완성되면 실제 합성 수행."""
        self.viewer.enable_selection(False)
        if self.current_image is None or self.pending_object_path is None:
            return
        use_corr = self.chk_use_correction.isChecked()
        out, defects = apply_object_at(
            self.current_image,
            self.object_dir,
            self.pending_object_path,
            x,
            y,
            w,
            h,
            0.0,
            use_corr,
        )
        self.current_image = out
        self.defects.extend(defects)
        self.viewer.set_image(cvimg_to_qpixmap(self.current_image))
        self.dirty = True

    def _on_object_dropped(self, path_str: str, cx: int, cy: int):
        """썸네일에서 드래그&드롭 → 임시 오버레이로만 표시 (적용 전까지 수정 가능)."""
        if self.current_image is None or not path_str:
            return

        object_path = Path(path_str)
        pix = QPixmap(str(object_path))
        if pix.isNull():
            return

        h_img, w_img, _ = self.current_image.shape
        # 기본 폭: 이미지 폭의 15%
        base_w = int(w_img * 0.15)
        if base_w <= 0:
            base_w = pix.width()
        aspect = pix.height() / pix.width() if pix.width() > 0 else 1.0
        target_w = max(8, min(base_w, w_img))
        target_h = int(target_w * aspect)

        x = int(cx - target_w / 2)
        y = int(cy - target_h / 2)
        x = max(0, min(w_img - target_w, x))
        y = max(0, min(h_img - target_h, y))

        self.overlay_object_path = object_path
        self.viewer.set_overlay(pix, QRect(x, y, target_w, target_h))

    def reset_image(self):
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.defects.clear()
        self.viewer.set_image(cvimg_to_qpixmap(self.current_image))
        self.dirty = False

    # ===== 자동 생성 로직 =====
    def run_auto_generation(self):
        """자동 탭 설정을 이용해 선택된 이미지(또는 전체)에 이물을 자동 합성."""
        if self.current_folder_path_auto is None:
            QMessageBox.warning(self, "경고", "먼저 자동 탭에서 이미지 폴더를 선택해 주세요.")
            return

        # 대상 원본 이미지 목록
        target_items = self.image_list_auto.selectedItems()
        if target_items:
            paths = [Path(item.data(Qt.UserRole)) for item in target_items if item.data(Qt.UserRole)]
        else:
            paths = []
            for i in range(self.image_list_auto.count()):
                item = self.image_list_auto.item(i)
                path_str = item.data(Qt.UserRole)
                if path_str:
                    paths.append(Path(path_str))

        if not paths:
            QMessageBox.warning(self, "경고", "자동으로 처리할 이미지가 없습니다.")
            return

        # 활성화된 이물 카테고리와 확률
        enabled_categories = []
        weights = []
        for name, btn in self.auto_category_checks.items():
            if btn.isChecked():
                prob = self.auto_category_probs[name].value()
                if prob > 0 and self.object_categories.get(name):
                    enabled_categories.append(name)
                    weights.append(prob)
        if not enabled_categories:
            QMessageBox.warning(self, "경고", "자동으로 사용할 이물 카테고리를 선택해 주세요.")
            return

        min_defects = self.auto_min_defects.value()
        max_defects = self.auto_max_defects.value()
        min_images = self.auto_min_images.value()
        max_images = self.auto_max_images.value()
        if min_defects > max_defects or min_images > max_images:
            QMessageBox.warning(self, "경고", "최소/최대 값이 올바르지 않습니다.")
            return

        for src_path in paths:
            img = imread_unicode(src_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            num_new = random.randint(min_images, max_images)
            for _ in range(num_new):
                aug = img.copy()
                defects: list[DefectInfo] = []

                num_defects = random.randint(min_defects, max_defects)
                h_img, w_img, _ = aug.shape

                for _d in range(num_defects):
                    # 카테고리 선택(확률 가중치)
                    cat = random.choices(enabled_categories, weights=weights, k=1)[0]
                    obj_paths = self.object_categories.get(cat, [])
                    if not obj_paths:
                        continue
                    obj_path = random.choice(obj_paths)

                    # 크기/위치/각도 랜덤 설정
                    base_ratio = random.uniform(0.08, 0.2)
                    base_w = int(w_img * base_ratio)
                    base_w = max(8, base_w)
                    # 가로세로 비율은 원본 이물 비율에 맞추도록 apply_object_at 안에서 처리됨
                    w = base_w
                    h = base_w

                    cx = random.randint(w // 2, max(w_img - w // 2, w // 2))
                    cy = random.randint(h // 2, max(h_img - h // 2, h // 2))
                    angle = random.uniform(-45.0, 45.0)

                    aug, new_defects = apply_object_at(
                        aug,
                        self.object_dir,
                        obj_path,
                        cx,
                        cy,
                        w,
                        h,
                        angle,
                    )
                    defects.extend(new_defects)

                # 자동 생성된 이미지 저장 (원본 이름 기준으로 번호 붙이기)
                stem = src_path.stem
                ext = src_path.suffix or ".png"
                folder = src_path.parent
                n = 1
                while True:
                    candidate = folder / f"{stem}_A{n:03d}{ext}"
                    if not candidate.exists():
                        save_path = candidate
                        break
                    n += 1

                if not imwrite_unicode(save_path, aug):
                    continue

                # 자동 탭 리스트에 추가
                item = QListWidgetItem(save_path.name)
                item.setData(Qt.UserRole, str(save_path))
                self.image_list_auto.addItem(item)

    # 저장
    def save_result(self, ask_if_no_base: bool = True) -> bool:
        """현재 이미지를 저장. 기본은 원본 이름 + 번호를 붙여 자동 저장."""
        if self.current_image is None:
            return False

        base_path = self.current_image_path
        save_path: str | Path | None = None

        if base_path is not None:
            folder = base_path.parent
            stem = base_path.stem
            ext = base_path.suffix or ".png"
            n = 1
            while True:
                candidate = folder / f"{stem}_{n:03d}{ext}"
                if not candidate.exists():
                    save_path = candidate
                    break
                n += 1
        elif ask_if_no_base:
            # 폴더/원본 정보가 없으면 기존 방식대로 경로를 묻는다.
            default_dir = ""
            save_path_str, _ = QFileDialog.getSaveFileName(
                self,
                "결함 이미지 저장",
                os.path.join(default_dir, "defect_image.png"),
                "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
            )
            if not save_path_str:
                return False
            save_path = save_path_str
        else:
            return False

        if not imwrite_unicode(save_path, self.current_image):
            return False

        # 상태 업데이트
        if isinstance(save_path, str):
            self.current_image_path = Path(save_path)
        else:
            self.current_image_path = save_path
        self.original_image = self.current_image.copy()
        self.dirty = False

        # 우측 파일 리스트에 반영
        if self.current_image_path is not None:
            self._add_image_to_list(self.current_image_path, select=True)

        return True


def main():
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


