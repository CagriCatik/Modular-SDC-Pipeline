"""PySide6 real-time dashboard for observing Modular SDC Pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import splev

from .interfaces import LaneDetections, ObserverStep, PipelineObserver


def _eval_lane_curve(boundary: Any, num_points: int = 24) -> Optional[np.ndarray]:
    """Convert either a (2, N) array or a splprep (t, c, k) tuple to a (2, N) array."""
    if boundary is None:
        return None
    if isinstance(boundary, np.ndarray) and boundary.ndim == 2 and boundary.shape[0] == 2:
        return boundary
    if isinstance(boundary, (tuple, list)) and len(boundary) == 3:
        try:
            u = np.linspace(0.0, 1.0, num_points)
            pts = np.asarray(splev(u, boundary), dtype=float)
            if pts.ndim == 2 and pts.shape[0] == 2 and np.isfinite(pts).all():
                return pts
        except Exception:
            return None
    return None


def _to_uint8_rgb(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation)
    if obs.ndim == 2:
        obs = np.stack([obs] * 3, axis=-1)
    if obs.shape[-1] == 1:
        obs = np.repeat(obs, 3, axis=-1)
    if obs.dtype != np.uint8:
        if obs.max(initial=1.0) <= 1.5:
            obs = np.clip(obs * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            obs = np.clip(obs, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(obs)


def _compute_edge_preview(observation: np.ndarray, cut_size: int, gradient_threshold: float = 14.0) -> np.ndarray:
    """Compute a high-contrast RGB edge map matching LaneDetection's cropped gradient."""
    rgb = _to_uint8_rgb(observation).astype(np.float32)
    h, w, _ = rgb.shape
    cut = min(max(cut_size, 2), h)
    gray = np.dot(rgb[:cut, :, :3], [0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    gsum = np.abs(gx) + np.abs(gy)
    gsum[gsum < gradient_threshold] = 0.0
    norm = np.clip((gsum / max(float(gsum.max()), 1.0)) * 255.0, 0.0, 255.0).astype(np.uint8)

    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:cut, :, 0] = norm
    out[:cut, :, 1] = np.clip(norm * 1.2, 0, 255).astype(np.uint8)
    out[:cut, :, 2] = 255 * (norm > 0).astype(np.uint8)
    return np.ascontiguousarray(out)


@dataclass
class PySideDashboard(PipelineObserver):
    """Native Qt6 (PySide6) dashboard for live perception, planning, and control telemetry."""

    max_history: int = 200
    cut_size: int = 68
    gradient_threshold: float = 14.0
    controller_name: str = "Lateral"
    window_title: str = "Modular SDC Pipeline — Live Observer"

    _qt: Any = field(init=False, repr=False)
    _app: Any = field(init=False, default=None, repr=False)
    _window: Any = field(init=False, default=None, repr=False)
    _paused: bool = field(init=False, default=False, repr=False)
    _episode_index: int = field(init=False, default=0, repr=False)
    _steps: List[float] = field(init=False, default_factory=list, repr=False)
    _speed_actual: List[float] = field(init=False, default_factory=list, repr=False)
    _speed_target: List[float] = field(init=False, default_factory=list, repr=False)
    _steer: List[float] = field(init=False, default_factory=list, repr=False)
    _gas: List[float] = field(init=False, default_factory=list, repr=False)
    _brake: List[float] = field(init=False, default_factory=list, repr=False)
    _rewards: List[float] = field(init=False, default_factory=list, repr=False)
    _cte: List[float] = field(init=False, default_factory=list, repr=False)
    _heading_deg: List[float] = field(init=False, default_factory=list, repr=False)
    _speed_err: List[float] = field(init=False, default_factory=list, repr=False)
    _episode_reward: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.max_history <= 0:
            raise ValueError("max_history must be positive")
        self._qt = self._import_pyside()

    def _import_pyside(self):
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except ImportError as exc:
            raise RuntimeError(
                "PySideDashboard requires PySide6 or PySide6-Essentials to be installed"
            ) from exc
        return QtCore, QtGui, QtWidgets

    def _ensure_window(self) -> None:
        if self._window is not None:
            return

        QtCore, QtGui, QtWidgets = self._qt

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        self._app = app

        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        fixed_font.setPointSize(12)

        class _ObservationCanvas(QtWidgets.QWidget):
            def __init__(self, title: str, overlay: bool = False) -> None:
                super().__init__()
                self.title = title
                self.overlay = overlay
                self.image: Optional[np.ndarray] = None
                self.left_pts: Optional[np.ndarray] = None
                self.right_pts: Optional[np.ndarray] = None
                self.waypoints: Optional[np.ndarray] = None
                self.cut_size: int = 68
                self.setMinimumSize(210, 170)

            def paintEvent(self, event) -> None:  # noqa: N802
                painter = QtGui.QPainter(self)
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                rect = self.rect()
                painter.fillRect(rect, QtGui.QColor("#11161d"))

                painter.setPen(QtGui.QColor("#cbd5e1"))
                font = painter.font()
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(10, 20, self.title)

                if self.image is None:
                    return

                h, w, _ = self.image.shape
                qimg = QtGui.QImage(
                    self.image.data, w, h, 3 * w, QtGui.QImage.Format_RGB888
                )
                pad_top = 28
                avail_w = max(rect.width() - 20, 1)
                avail_h = max(rect.height() - pad_top - 10, 1)
                side = min(avail_w, avail_h)
                x0 = (rect.width() - side) // 2
                y0 = pad_top + (avail_h - side) // 2
                target_rect = QtCore.QRect(x0, y0, side, side)
                painter.drawImage(target_rect, qimg)

                if not self.overlay:
                    return

                scale_x = side / max(w - 1, 1)
                scale_y = side / max(h - 1, 1)

                def to_px(x_crop: float, y_crop: float) -> QtCore.QPointF:
                    row_obs = (self.cut_size - 1) - y_crop
                    return QtCore.QPointF(x0 + x_crop * scale_x, y0 + row_obs * scale_y)

                horizon_y = y0 + (self.cut_size - 1) * scale_y
                painter.setPen(QtGui.QPen(QtGui.QColor("#64748b"), 1, QtCore.Qt.DashLine))
                painter.drawLine(QtCore.QPointF(x0, horizon_y), QtCore.QPointF(x0 + side, horizon_y))

                # Center reference line
                cx = x0 + 48.0 * scale_x
                painter.setPen(QtGui.QPen(QtGui.QColor("#475569"), 1, QtCore.Qt.DotLine))
                painter.drawLine(QtCore.QPointF(cx, y0), QtCore.QPointF(cx, horizon_y))

                for pts, color in (
                    (self.left_pts, "#38bdf8"),
                    (self.right_pts, "#fb923c"),
                ):
                    if pts is not None and pts.shape[1] >= 2:
                        painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.5))
                        poly = QtGui.QPolygonF(
                            [to_px(float(pts[0, i]), float(pts[1, i])) for i in range(pts.shape[1])]
                        )
                        painter.drawPolyline(poly)

                if (
                    self.waypoints is not None
                    and self.waypoints.ndim == 2
                    and self.waypoints.shape[0] == 2
                    and self.waypoints.shape[1] >= 1
                ):
                    painter.setPen(QtGui.QPen(QtGui.QColor("#4ade80"), 2.0))
                    poly = QtGui.QPolygonF(
                        [
                            to_px(float(self.waypoints[0, i]), float(self.waypoints[1, i]))
                            for i in range(self.waypoints.shape[1])
                        ]
                    )
                    painter.drawPolyline(poly)
                    painter.setBrush(QtGui.QColor("#4ade80"))
                    for pt in poly:
                        painter.drawEllipse(pt, 3.5, 3.5)

        class _SeriesCanvas(QtWidgets.QWidget):
            def __init__(
                self,
                title: str,
                series_specs: Sequence[Tuple[str, str]],
                fixed_ylim: Optional[Tuple[float, float]] = None,
            ) -> None:
                super().__init__()
                self.title = title
                self.series_specs = list(series_specs)
                self.fixed_ylim = fixed_ylim
                self.data: List[List[float]] = [[] for _ in self.series_specs]
                self.setMinimumSize(210, 150)

            def paintEvent(self, event) -> None:  # noqa: N802
                painter = QtGui.QPainter(self)
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                rect = self.rect()
                painter.fillRect(rect, QtGui.QColor("#11161d"))

                painter.setPen(QtGui.QColor("#cbd5e1"))
                font = painter.font()
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(10, 20, self.title)

                font.setBold(False)
                painter.setFont(font)
                lx = rect.width() - 10
                for name, color in reversed(self.series_specs):
                    tw = painter.fontMetrics().horizontalAdvance(name) + 18
                    lx -= tw
                    painter.setPen(QtGui.QColor(color))
                    painter.drawLine(lx, 15, lx + 10, 15)
                    painter.setPen(QtGui.QColor("#cbd5e1"))
                    painter.drawText(lx + 14, 20, name)

                left, top, right, bottom = 46, 34, rect.width() - 14, rect.height() - 18
                pw = max(right - left, 1)
                ph = max(bottom - top, 1)

                painter.setPen(QtGui.QPen(QtGui.QColor("#1e293b"), 1))
                painter.drawRect(left, top, pw, ph)
                painter.drawLine(left, top + ph // 2, right, top + ph // 2)

                if self.fixed_ylim is not None:
                    ymin, ymax = self.fixed_ylim
                else:
                    flat = [v for s in self.data for v in s if np.isfinite(v)]
                    ymin = min(0.0, min(flat)) if flat else 0.0
                    ymax = max(5.0, max(flat) * 1.1) if flat else 35.0
                if ymax <= ymin:
                    ymax = ymin + 1.0

                painter.setPen(QtGui.QColor("#94a3b8"))
                painter.drawText(4, top + 10, f"{ymax:.1f}")
                painter.drawText(4, bottom, f"{ymin:.1f}")

                for values, (_, color) in zip(self.data, self.series_specs):
                    n = len(values)
                    if n < 2:
                        continue
                    painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.0))
                    pts = []
                    for idx, val in enumerate(values):
                        if not np.isfinite(val):
                            continue
                        x = left + (idx / max(n - 1, 1)) * pw
                        y = bottom - ((val - ymin) / (ymax - ymin)) * ph
                        y = float(np.clip(y, top, bottom))
                        pts.append(QtCore.QPointF(x, y))
                    if len(pts) >= 2:
                        painter.drawPolyline(QtGui.QPolygonF(pts))

        window = QtWidgets.QMainWindow()
        window.setWindowTitle(self.window_title)
        window.resize(1040, 620)
        window.setStyleSheet("QMainWindow { background-color: #0b0f14; color: #e2e8f0; }")

        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # Top telemetry & controls bar
        top_bar = QtWidgets.QHBoxLayout()
        telemetry_label = QtWidgets.QLabel("Waiting for episode reset...")
        telemetry_label.setFont(fixed_font)
        telemetry_label.setStyleSheet(
            "background-color: #161d26; color: #e2e8f0; padding: 6px 10px; border-radius: 6px;"
        )
        pause_btn = QtWidgets.QPushButton("Pause")
        pause_btn.setStyleSheet(
            "QPushButton { background-color: #1e293b; color: #e2e8f0; padding: 6px 12px; "
            "border-radius: 6px; font-weight: bold; } "
            "QPushButton:hover { background-color: #334155; }"
        )

        def _toggle_pause() -> None:
            self._paused = not self._paused
            pause_btn.setText("Resume" if self._paused else "Pause")

        pause_btn.clicked.connect(_toggle_pause)
        top_bar.addWidget(telemetry_label, stretch=1)
        top_bar.addWidget(pause_btn)
        root_layout.addLayout(top_bar)

        # Compact 2x4 grid of pipeline debug cards
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)

        obs_canvas = _ObservationCanvas("1. Raw Observation", overlay=False)
        edge_canvas = _ObservationCanvas("2. Perception: Gradient Edges", overlay=False)
        overlay_canvas = _ObservationCanvas("3. Lanes & Planned Waypoints", overlay=True)
        overlay_canvas.cut_size = self.cut_size

        lateral_canvas = _SeriesCanvas(
            "4. Lateral CTE [px] & Heading [deg]",
            [("CTE", "#f472b6"), ("Heading", "#fbbf24")],
        )
        speed_canvas = _SeriesCanvas(
            "5. Speed Tracking [m/s]",
            [("Actual", "#4ade80"), ("Target", "#38bdf8")],
        )
        speed_err_canvas = _SeriesCanvas(
            "6. Speed Error (Tgt - Act)",
            [("Error", "#fb7185")],
        )
        control_canvas = _SeriesCanvas(
            "7. Control Commands",
            [("Steer", "#f59e0b"), ("Gas", "#4ade80"), ("Brake", "#f87171")],
            fixed_ylim=(-1.05, 1.05),
        )
        reward_canvas = _SeriesCanvas(
            "8. Cumulative Return",
            [("Return", "#c084fc")],
        )

        grid.addWidget(obs_canvas, 0, 0)
        grid.addWidget(edge_canvas, 0, 1)
        grid.addWidget(overlay_canvas, 0, 2)
        grid.addWidget(lateral_canvas, 0, 3)
        grid.addWidget(speed_canvas, 1, 0)
        grid.addWidget(speed_err_canvas, 1, 1)
        grid.addWidget(control_canvas, 1, 2)
        grid.addWidget(reward_canvas, 1, 3)
        root_layout.addLayout(grid, stretch=1)

        window.setCentralWidget(central)
        window.show()
        window.raise_()
        window.activateWindow()

        self._window = window
        self._telemetry_label = telemetry_label
        self._obs_canvas = obs_canvas
        self._edge_canvas = edge_canvas
        self._overlay_canvas = overlay_canvas
        self._lateral_canvas = lateral_canvas
        self._speed_canvas = speed_canvas
        self._speed_err_canvas = speed_err_canvas
        self._control_canvas = control_canvas
        self._reward_canvas = reward_canvas

    def _append(self, buffer: List[float], value: float) -> None:
        buffer.append(value)
        if len(buffer) > self.max_history:
            del buffer[0]

    def on_reset(self, observation: np.ndarray, info: Mapping[str, Any]) -> None:  # type: ignore[override]
        self._ensure_window()
        self._episode_index += 1
        self._steps.clear()
        self._speed_actual.clear()
        self._speed_target.clear()
        self._steer.clear()
        self._gas.clear()
        self._brake.clear()
        self._rewards.clear()
        self._cte.clear()
        self._heading_deg.clear()
        self._speed_err.clear()
        self._episode_reward = 0.0

        rgb = _to_uint8_rgb(observation)
        edges = _compute_edge_preview(observation, self.cut_size, self.gradient_threshold)
        self._obs_canvas.image = rgb
        self._edge_canvas.image = edges
        self._overlay_canvas.image = rgb
        self._overlay_canvas.left_pts = None
        self._overlay_canvas.right_pts = None
        self._overlay_canvas.waypoints = None
        self._lateral_canvas.data = [[], []]
        self._speed_canvas.data = [[], []]
        self._speed_err_canvas.data = [[]]
        self._control_canvas.data = [[], [], []]
        self._reward_canvas.data = [[]]
        self._telemetry_label.setText(
            f"Ep {self._episode_index} | Controller: {self.controller_name} | "
            f"speed={float(info.get('speed', 0.0)):5.2f} m/s"
        )
        if self._window is not None:
            self._window.raise_()
            self._window.activateWindow()
        self._refresh()

    def on_step(self, step: ObserverStep) -> None:  # type: ignore[override]
        self._ensure_window()
        while self._paused and self._window is not None and self._window.isVisible():
            self._app.processEvents()

        self._episode_reward += float(step.reward)

        rgb = _to_uint8_rgb(step.observation)
        edges = _compute_edge_preview(step.observation, self.cut_size, self.gradient_threshold)
        lanes: Optional[LaneDetections] = step.perception.lanes
        left_pts = _eval_lane_curve(lanes.left_boundary) if lanes is not None else None
        right_pts = _eval_lane_curve(lanes.right_boundary) if lanes is not None else None

        self._obs_canvas.image = rgb
        self._edge_canvas.image = edges
        self._overlay_canvas.image = rgb
        self._overlay_canvas.left_pts = left_pts
        self._overlay_canvas.right_pts = right_pts
        self._overlay_canvas.waypoints = step.planning.waypoints

        actual_speed = float(step.info.get("speed", step.context.speed))
        target_speed = (
            float(step.planning.target_speed)
            if step.planning.target_speed is not None
            else float("nan")
        )
        speed_err = (target_speed - actual_speed) if np.isfinite(target_speed) else 0.0

        cte = 0.0
        heading_deg = 0.0
        wp = step.planning.waypoints
        if wp is not None and wp.ndim == 2 and wp.shape[0] == 2 and wp.shape[1] >= 1:
            cte = 48.0 - float(wp[0, 0])
            if wp.shape[1] >= 2:
                dx = float(wp[1, 1] - wp[1, 0])
                dy = float(wp[0, 0] - wp[0, 1])
                heading_deg = float(np.degrees(np.arctan2(dy, dx))) if (abs(dx) + abs(dy)) > 1e-6 else 0.0

        lane_status = "OK" if (left_pts is not None and right_pts is not None) else "FALLBACK"

        self._append(self._steps, float(step.context.step_index))
        self._append(self._speed_actual, actual_speed)
        self._append(self._speed_target, target_speed)
        self._append(self._speed_err, speed_err)
        self._append(self._cte, cte)
        self._append(self._heading_deg, heading_deg)
        self._append(self._steer, float(step.command.steer))
        self._append(self._gas, float(step.command.gas))
        self._append(self._brake, float(step.command.brake))
        self._append(self._rewards, self._episode_reward)

        self._lateral_canvas.data = [self._cte, self._heading_deg]
        self._speed_canvas.data = [self._speed_actual, self._speed_target]
        self._speed_err_canvas.data = [self._speed_err]
        self._control_canvas.data = [self._steer, self._gas, self._brake]
        self._reward_canvas.data = [self._rewards]

        self._telemetry_label.setText(
            f"Ep {self._episode_index} | Step {step.context.step_index:4d} | "
            f"{self.controller_name} | Lanes: {lane_status} | "
            f"CTE: {cte:+5.1f}px | Hdg: {heading_deg:+5.1f}° | "
            f"v: {actual_speed:4.1f}/{target_speed:4.1f} | "
            f"u: ({step.command.steer:+.2f}, {step.command.gas:.2f}, {step.command.brake:.2f}) | "
            f"Ret: {self._episode_reward:+.1f}"
        )
        self._refresh()

    def _refresh(self) -> None:
        if self._window is None:
            return
        self._obs_canvas.update()
        self._edge_canvas.update()
        self._overlay_canvas.update()
        self._lateral_canvas.update()
        self._speed_canvas.update()
        self._speed_err_canvas.update()
        self._control_canvas.update()
        self._reward_canvas.update()
        if self._app is not None:
            self._app.processEvents()
