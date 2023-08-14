import os
import numpy as np
import matplotlib.pyplot as plt
import atexit

from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation
from collections import deque
from scipy.optimize import minimize


def delete_file(file):
    if os.path.exists(file):
      os.remove(file)
    else:
      print(file)


class PointOfApp:

    def __init__(self, file):

        self.file = file

        # Rotation and Translation matrices to find pos on mTMS head
        self.R = np.load('robot/resources/Rot.npy')
        self.T = np.load('robot/resources/Tr.npy')

        # Set tool origin and size bounds
        # ..... based on size of mTMS
        self.orig = np.array([0.0, 0.0, 0.05])
        self.bounds_x = (-0.15, 0.15)        
        self.bounds_y = (-0.15, 0.15)
        self.bounds_z = (0, 0.05)

        # for smoothing
        self.F_values = deque(maxlen=6)
        self.M_values = deque(maxlen=6)
        self.r_values = deque(maxlen=20)

        # create plots for point of application
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.square = FancyBboxPatch((-12, -12), 24, 24, alpha=0.5, boxstyle="round,pad=3")
        self.ax.add_patch(self.square)
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.point, = self.ax.plot(0, 0, 'ro', markersize=10)
        # self.kf_point, = self.ax2.plot(0, 0, 'g+', markersize=20)

        ani = animation.FuncAnimation(self.fig, self.loop, fargs=(), interval=1)
        plt.show()

        atexit.register(delete_file, self.file)

        #self.loop()

    def _func(self, r, F, M, orig):
        '''
        Function to minimise objective function

        Returns: Norm of r x F - M (Np Array)  
        '''
        # Check if r is outside the box
        return np.linalg.norm(np.cross(r - orig, F) - M)

    def find_r(self, F, M):
        '''
        Find point of application of force

        Returns: Minimised point of application (Np Array) 
        '''

        # initial guess
        r0 = np.array([0.0, 0.0, 0.05])

        # find r that minimizes the objective function
        res = minimize(self._func, r0, 
                       args=(F, M, self.orig), 
                       method='Nelder-Mead', 
                       bounds=(self.bounds_x, self.bounds_y, self.bounds_z))
        
        r_min = res.x * 100  # multiply by 100 to get value in cm

        return [round(r_min[i], 1) for i in range(0, len(r_min))]


    def loop(self, i):
        '''
        Loop to read live sensor data and perform relevant operations. 

        Performs: Live plotting, 
                write-to-csv, 
                normalise(tare) values
        '''
        with open(self.file, 'rb') as f:
            try:  # catch OSError in case of a one line file 
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            data = f.readline().decode()
            print(data)

        F_vect  = data[:3]
        M_vect = data[3:]

        # Add new value to the lists
        self.F_values.append(F_vect)
        self.M_values.append(M_vect)

        # Compute the average of the last N values
        F_avg = np.mean(self.F_values, axis=0)
        M_avg = np.mean(self.M_values, axis=0)

        r = self.find_r(F_avg, M_avg)
        # rotating R to the correct coordinates
        r_tran = self.R @ r + self.T
        r_tran = [round(r_tran[i], 1) for i in range(0, len(r_tran))]

        if not (-15 <= r_tran[0] <= 15 and 
                -15 <= r_tran[1] <= 15):
            self.point.set_data([0], [0])

        else:
            if F_avg[2] < -1: 
                self.point.set_data(r_tran[0], r_tran[1])
            else:
                self.point.set_data([0], [0])

        # Redraw the plot
        self.fig.canvas.draw()

