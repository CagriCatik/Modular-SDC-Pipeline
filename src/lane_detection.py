import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
import time

class LaneDetection:
    '''
    Lane detection module using edge detection and B-spline fitting

    Args:
        cut_size (int): Cut the image at the front of the car (default=68).
        spline_smoothness (float): Smoothness factor for spline fitting (default=10).
        gradient_threshold (float): Threshold for gradient magnitude (default=14).
        distance_maxima_gradient (int): Minimum distance between maxima in gradient (default=3).
    '''

    def __init__(self, cut_size=68, spline_smoothness=10, gradient_threshold=14, distance_maxima_gradient=3):
        self.car_position = np.array([48, 0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = None
        self.lane_boundary2_old = None

    def cut_gray(self, state_image_full):
        '''
        Cuts the image at the front end of the car and converts it to grayscale.

        Input:
            state_image_full (numpy.ndarray): 96x96x3 image.

        Output:
            numpy.ndarray: Grayscale image of size cut_size x 96 x 1.
        '''
        # Cut the image at the front end of the car
        gray_image = state_image_full[:self.cut_size]
        # Convert to grayscale
        gray_image = np.dot(gray_image[..., :3], [0.299, 0.587, 0.114])
        # Expand dimensions to maintain the shape (cut_size, 96, 1)
        gray_image = np.expand_dims(gray_image, axis=2)
        # Reverse the image vertically
        return gray_image[::-1]

    def edge_detection(self, gray_image):
        '''
        Performs edge detection by computing the absolute gradients and thresholding.

        Input:
            gray_image (numpy.ndarray): Grayscale image of size cut_size x 96 x 1.

        Output:
            numpy.ndarray: Gradient sum of size cut_size x 96 x 1.
        '''
        # Remove the singleton dimension
        gray_image = gray_image.squeeze()
        # Compute gradients along x and y axes
        grad_y, grad_x = np.gradient(gray_image)
        # Compute absolute gradients
        abs_grad_x = np.abs(grad_x)
        abs_grad_y = np.abs(grad_y)
        # Sum the absolute gradients
        gradient_sum = abs_grad_x + abs_grad_y
        # Thresholding
        gradient_sum[gradient_sum < self.gradient_threshold] = 0
        # Expand dimensions to maintain the shape (cut_size, 96, 1)
        gradient_sum = np.expand_dims(gradient_sum, axis=2)
        return gradient_sum

    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        Finds local maxima for each row of the gradient image.

        Input:
            gradient_sum (numpy.ndarray): Gradient sum of size cut_size x 96 x 1.

        Output:
            numpy.ndarray: 2 x Number of maxima array containing column and row indices.
        '''
        maxima_list = []
        for row_index in range(gradient_sum.shape[0]):
            row = gradient_sum[row_index, :, 0]
            # Find peaks in the row with the specified minimum distance
            peaks, _ = find_peaks(row, distance=self.distance_maxima_gradient)
            for col_index in peaks:
                maxima_list.append([col_index, row_index])
        # Convert to numpy array and transpose to get shape (2, Number of maxima)
        argmaxima = np.array(maxima_list).T
        return argmaxima

    def find_first_lane_point(self, gradient_sum):
        '''
        Finds the first lane boundary points above the car.

        Input:
            gradient_sum (numpy.ndarray): Gradient sum of size cut_size x 96 x 1.

        Output:
            tuple: (lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found)
        '''
        lanes_found = False
        row = 0

        # Loop through the rows
        while not lanes_found and row < self.cut_size:
            # Find peaks with minimum distance
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
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]], row]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]], row]])
                lanes_found = True

            row += 1

            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0, 0]])
                lane_boundary2_startpoint = np.array([[0, 0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found

    def lane_detection(self, state_image_full):
        '''
        Performs the road detection.

        Args:
            state_image_full (numpy.ndarray): Image of size 96 x 96 x 3.

        Returns:
            tuple: (lane_boundary1 spline, lane_boundary2 spline)
        '''
        # Convert to grayscale and cut the image
        gray_state = self.cut_gray(state_image_full)

        # Edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        # Find first lane boundary points
        lane_boundary1_startpoint, lane_boundary2_startpoint, lane_found = self.find_first_lane_point(gradient_sum)

        if lane_found:
            # Convert maxima to a list of [x, y]
            maxima_list = maxima.T.tolist()

            # Initialize lane boundary points with the starting points
            lane_boundary1_points = [lane_boundary1_startpoint[0].tolist()]
            lane_boundary2_points = [lane_boundary2_startpoint[0].tolist()]

            # Remove starting points from maxima_list if they exist
            for lane_point in [lane_boundary1_startpoint[0], lane_boundary2_startpoint[0]]:
                if any(np.array_equal(lane_point, maxima) for maxima in maxima_list):
                    maxima_list.remove(lane_point.tolist())  # Convert to list to match the maxima_list format


            # Function to find lane points
            def find_lane_points(lane_points):
                last_point = lane_points[-1]
                while True:
                    next_row = last_point[1] + 1
                    if next_row >= self.cut_size:
                        break
                    # Get maxima in the next row
                    maxima_in_row = [p for p in maxima_list if p[1] == next_row]
                    if not maxima_in_row:
                        break
                    # Find the maximum with the lowest distance to the last lane point
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

            # Find lane boundary points
            lane_boundary1_points = find_lane_points(lane_boundary1_points)
            lane_boundary2_points = find_lane_points(lane_boundary2_points)

            # Convert lists to numpy arrays
            lane_boundary1_points = np.array(lane_boundary1_points)
            lane_boundary2_points = np.array(lane_boundary2_points)

            # Spline fitting using scipy.interpolate.splprep
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
                # Lane boundary 1
                tck1, _ = splprep([lane_boundary1_points[:, 0], lane_boundary1_points[:, 1]], s=self.spline_smoothness)
                lane_boundary1 = tck1
                # Lane boundary 2
                tck2, _ = splprep([lane_boundary2_points[:, 0], lane_boundary2_points[:, 1]], s=self.spline_smoothness)
                lane_boundary2 = tck2
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        return lane_boundary1, lane_boundary2

    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plots lanes and waypoints.

        Args:
            state_image_full (numpy.ndarray): Image of size 96 x 96 x 3.
            steps (int): Current step in the simulation or process.
            fig (matplotlib.figure.Figure): Matplotlib figure object.
            waypoints (list): List of waypoints to plot.
        '''
        # Evaluate spline for 6 different spline parameters
        t = np.linspace(0, 1, 6)
        if self.lane_boundary1_old is not None and self.lane_boundary2_old is not None:
            lane_boundary1_points = np.array(splev(t, self.lane_boundary1_old))
            lane_boundary2_points = np.array(splev(t, self.lane_boundary2_old))
        else:
            lane_boundary1_points = np.zeros((2, 6))
            lane_boundary2_points = np.zeros((2, 6))

        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points[0], lane_boundary1_points[1] + 96 - self.cut_size, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points[0], lane_boundary2_points[1] + 96 - self.cut_size, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1] + 96 - self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5, 95.5))
        plt.ylim((-0.5, 95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
