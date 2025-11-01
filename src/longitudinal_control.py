import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

class LongitudinalController:
    '''
    Longitudinal Control using a PID Controller

    functions:
        PID_step()
        control()
    '''
    def __init__(self, KP=0.01, KI=0.0, KD=0.0, integral_windup_limit=10):
        self.last_error = 0
        self.sum_error = 0
        self.last_control = 0
        self.speed_history = []
        self.target_speed_history = []
        self.step_history = [] 

        # PID parameters
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.integral_windup_limit = integral_windup_limit

    def PID_step(self, speed, target_speed, dt=1.0):
        '''
        Perform one step of the PID control
        - Implement the discretized control law.
        - Implement a maximum value for the sum of error you are using for the integral term 

        args: 
            speed: current vehicle speed
            target_speed: desired target speed
            dt: time step (default = 1.0 for simplicity)

        output: 
            control (u): the control signal which dictates gas or brake values
        '''
        
        # 1. Error Calculation
        error = target_speed - speed

        # 2. Proportional Term
        P_term = self.KP * error

        # 3. Integral Term
        self.sum_error += error * dt  # sum of errors over time
        # Prevent integral windup by clamping the sum_error
        self.sum_error = np.clip(self.sum_error, -self.integral_windup_limit, self.integral_windup_limit)
        I_term = self.KI * self.sum_error

        # 4. Derivative Term
        D_term = self.KD * (error - self.last_error) / dt

        # 5. Compute Control Signal (PID Output)
        control = P_term + I_term + D_term

        # 6. Store the last error for the next derivative calculation
        self.last_error = error

        return control

    def control(self, speed, target_speed):
        '''
        Derive action values for gas and brake via the control signal
        using PID controlling

        Args:
            speed (float): Current speed of the vehicle
            target_speed (float): Desired target speed of the vehicle

        output:
            gas (float): Gas throttle value between 0 and 0.8
            brake (float): Brake value between 0 and 0.8
        '''

        # Get the control signal from the PID controller
        control = self.PID_step(speed, target_speed)
        brake = 0
        gas = 0

        # Translate the control signal to gas and brake actions
        if control >= 0:
            gas = np.clip(control, 0, 0.8)  # Throttle should be between 0 and 0.8
        else:
            brake = np.clip(-1 * control, 0, 0.8)  # Brake should be between 0 and 0.8

        return gas, brake

    def plot_speed(self, speed, target_speed, step, fig):
        '''
        Plot the speed history and target speed history for visualization.
        '''
        self.speed_history.append(speed)
        self.target_speed_history.append(target_speed)
        self.step_history.append(step)
        plt.gcf().clear()
        plt.plot(self.step_history, self.speed_history, c="green")
        plt.plot(self.step_history, self.target_speed_history)
        fig.canvas.flush_events()
