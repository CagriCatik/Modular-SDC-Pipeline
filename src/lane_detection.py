import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from collections import OrderedDict
import time


class LaneDetection:
    """
    Lane detection module using edge detection and B-spline fitting.

    Args:
        cut_size (int): Cut the image at the front of the car (default=68).
        spline_smoothness (float): Smoothness factor for spline fitting (default=10).
        gradient_threshold (float): Threshold for gradient magnitude (default=14).
        distance_maxima_gradient (int): Minimum distance between maxima in gradient (default=3).
        debug (bool): If True, collect intermediate frames in named buffers.
    """

    def __init__(self, cut_size=68, spline_smoothness=10, gradient_threshold=14, distance_maxima_gradient=3, debug=False):
        self.car_position = np.array([48, 0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = None
        self.lane_boundary2_old = None

        # Debug
        self.debug = debug
        self._debug_frames = OrderedDict()

    # -------------------------
    # Debug helpers
    # -------------------------
    def _dbg(self, name: str, img: np.ndarray):
        if not self.debug:
            return
        self._debug_frames[name] = np.asarray(img)

    def get_debug_frames(self, reset: bool = True):
        out = dict(self._debug_frames)
        if reset:
            self._debug_frames.clear()
        return out

    # -------------------------
    # Core pipeline
    # -------------------------
    def cut_gray(self, state_image_full):
        """
        Cuts the image at the front end of the car and converts it to grayscale.

        Input:
            state_image_full (numpy.ndarray): 96x96x3 image.

        Output:
            numpy.ndarray: Grayscale image of size cut_size x 96 x 1.
        """
        gray_image = state_image_full[:self.cut_size]
        gray_image = np.dot(gray_image[..., :3], [0.299, 0.587, 0.114])
        gray_image = np.expand_dims(gray_image, axis=2)
        return gray_image[::-1]

    def edge_detection(self, gray_image):
        """
        Performs edge detection by computing the absolute gradients and thresholding.

        Input:
            gray_image (numpy.ndarray): Grayscale image of size cut_size x 96 x 1.

        Output:
            numpy.ndarray: Gradient sum of size cut_size x 96 x 1.
        """
        gray2d = gray_image.squeeze()
        grad_y, grad_x = np.gradient(gray2d)
        abs_grad_x = np.abs(grad_x)
        abs_grad_y = np.abs(grad_y)
        gradient_sum = abs_grad_x + abs_grad_y
        gradient_sum = np.where(gradient_sum < self.gradient_threshold, 0.0, gradient_sum)
        gradient_sum = np.expand_dims(gradient_sum, axis=2)
        return gradient_sum

    def find_maxima_gradient_rowwise(self, gradient_sum):
        """
        Finds local maxima for each row of the gradient image.

        Input:
            gradient_sum (numpy.ndarray): Gradient sum of size cut_size x 96 x 1.

        Output:
            numpy.ndarray: 2 x Number of maxima array containing column and row indices.
        """
        maxima_list = []
        for row_index in range(gradient_sum.shape[0]):
            row = gradient_sum[row_index, :, 0]
            peaks, _ = find_peaks(row, distance=self.distance_maxima_gradient)
            for col_index in peaks:
                maxima_list.append([col_index, row_index])
        if len(maxima_list) == 0:
            return np.zeros((2, 0), dtype=int)
        argmaxima = np.array(maxima_list).T
        return argmaxima

    def find_first_lane_point(self, gradient_sum):
        """
        Finds the first lane boundary points above the car.

        Input:
            gradient_sum (numpy.ndarray): Gradient sum of size cut_size x 96 x 1.

        Output:
            tuple: (lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found)
        """
        lanes_found = False
        row = 0

        while not lanes_found and row < self.cut_size:
            argmaxima = find_peaks(gradient_sum[row, :, 0], distance=3)[0]

            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0], row]])
                lane_boundary2_startpoint = np.array([[0 if argmaxima[0] >= 48 else 95, row]])
                lanes_found = True

            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0], row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1], row]])
                lanes_found = True

            elif argmaxima.shape[0] > 2:
                A = np.argsort((argmaxima - self.car_position[0]) ** 2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]], row]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]], row]])
                lanes_found = True

            row += 1

            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0, 0]])
                lane_boundary2_startpoint = np.array([[0, 0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found

    def _maxima_mask_image(self, maxima, height, width):
        """
        Builds a binary visualization image (H x W) with maxima marked.
        """
        mask = np.zeros((height, width), dtype=np.float32)
        if maxima.size == 0:
            return mask
        cols = maxima[0, :]
        rows = maxima[1, :]
        cols = np.clip(cols, 0, width - 1)
        rows = np.clip(rows, 0, height - 1)
        mask[rows, cols] = 255.0
        return mask

    # -------------------------
    # Overlay helpers
    # -------------------------
    def _draw_disk(self, img, cx, cy, radius, color):
        h, w = img.shape[:2]
        x0 = max(0, cx - radius)
        x1 = min(w - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(h - 1, cy + radius)
        for y in range(y0, y1 + 1):
            yy = y - cy
            yy2 = yy * yy
            for x in range(x0, x1 + 1):
                xx = x - cx
                if xx * xx + yy2 <= radius * radius:
                    img[y, x] = color

    def _overlay_image_from_splines(self, state_image_full, lane1, lane2, thickness=3):
        """
        Returns an RGB image showing state_image_full with lane splines drawn on it.
        Coordinates match plot_state_lane.
        """
        img = np.ascontiguousarray(state_image_full[::-1].copy())
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        orange = np.array([255, 165, 0], dtype=np.uint8)

        def draw_lane(tck):
            if tck is None:
                return
            t = np.linspace(0, 1, 60)
            x, y = splev(t, tck)
            y = y + 96 - self.cut_size
            x = np.asarray(np.round(x), dtype=int)
            y = np.asarray(np.round(y), dtype=int)
            for xi, yi in zip(x, y):
                if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                    self._draw_disk(img, xi, yi, thickness, orange)

        draw_lane(lane1)
        draw_lane(lane2)
        return img

    # -------------------------
    # Pipeline entry
    # -------------------------
    def lane_detection(self, state_image_full):
        """
        Performs the road detection.

        Args:
            state_image_full (numpy.ndarray): Image of size 96 x 96 x 3.

        Returns:
            tuple: (lane_boundary1 spline, lane_boundary2 spline)
        """
        # self._dbg("00_input_rgb", state_image_full)  # disabled by request

        # Grayscale and crop
        gray_state = self.cut_gray(state_image_full)
        self._dbg("10_gray", gray_state.squeeze())

        # Edge map
        gradient_sum = self.edge_detection(gray_state)
        self._dbg("30_edges", gradient_sum.squeeze())

        # Rowwise maxima
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)
        maxima_img = self._maxima_mask_image(maxima, gradient_sum.shape[0], gradient_sum.shape[1])
        self._dbg("50_maxima_mask", maxima_img)

        # Find first points
        lane_boundary1_startpoint, lane_boundary2_startpoint, lane_found = self.find_first_lane_point(gradient_sum)

        if lane_found:
            maxima_list = maxima.T.tolist()
            lane_boundary1_points = [lane_boundary1_startpoint[0].tolist()]
            lane_boundary2_points = [lane_boundary2_startpoint[0].tolist()]

            for lane_point in [lane_boundary1_startpoint[0], lane_boundary2_startpoint[0]]:
                if any(np.array_equal(lane_point, m) for m in maxima_list):
                    maxima_list.remove(lane_point.tolist())

            def find_lane_points(lane_points):
                last_point = lane_points[-1]
                while True:
                    next_row = last_point[1] + 1
                    if next_row >= self.cut_size:
                        break
                    maxima_in_row = [p for p in maxima_list if p[1] == next_row]
                    if not maxima_in_row:
                        break
                    distances = [abs(p[0] - last_point[0]) for p in maxima_in_row]
                    min_distance = min(distances)
                    if min_distance >= 100:
                        break
                    min_index = distances.index(min_distance)
                    next_point = maxima_in_row[min_index]
                    lane_points.append(next_point)
                    maxima_list.remove(next_point)
                    last_point = next_point
                return lane_points

            lane_boundary1_points = find_lane_points(lane_boundary1_points)
            lane_boundary2_points = find_lane_points(lane_boundary2_points)

            lane_boundary1_points = np.array(lane_boundary1_points)
            lane_boundary2_points = np.array(lane_boundary2_points)

            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
                tck1, _ = splprep([lane_boundary1_points[:, 0], lane_boundary1_points[:, 1]], s=self.spline_smoothness)
                lane_boundary1 = tck1
                tck2, _ = splprep([lane_boundary2_points[:, 0], lane_boundary2_points[:, 1]], s=self.spline_smoothness)
                lane_boundary2 = tck2
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        # Persist last known splines
        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # Add overlay debug frame instead of 00_input_rgb
        overlay_img = self._overlay_image_from_splines(state_image_full, lane_boundary1, lane_boundary2, thickness=3)
        self._dbg("90_overlay", overlay_img)

        return lane_boundary1, lane_boundary2

    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        """
        Plot lanes and waypoints into the provided Matplotlib Figure (fig).
        """
        t = np.linspace(0, 1, 6)
        if self.lane_boundary1_old is not None and self.lane_boundary2_old is not None:
            lane_boundary1_points = np.array(splev(t, self.lane_boundary1_old))
            lane_boundary2_points = np.array(splev(t, self.lane_boundary2_old))
        else:
            lane_boundary1_points = np.zeros((2, 6))
            lane_boundary2_points = np.zeros((2, 6))

        fig.clf()
        ax = fig.add_subplot(111)

        ax.imshow(state_image_full[::-1])

        ax.plot(
            lane_boundary1_points[0],
            lane_boundary1_points[1] + 96 - self.cut_size,
            linewidth=5,
            color="orange",
        )
        ax.plot(
            lane_boundary2_points[0],
            lane_boundary2_points[1] + 96 - self.cut_size,
            linewidth=5,
            color="orange",
        )

        if len(waypoints):
            ax.scatter(
                waypoints[0],
                waypoints[1] + 96 - self.cut_size,
                c="white",
                s=20,
            )

        ax.set_xlim(-0.5, 95.5)
        ax.set_ylim(-0.5, 95.5)
        ax.axis("off")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
