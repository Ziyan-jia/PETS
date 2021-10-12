""" Generate data both for training and testing """
# Import necessary libraries
import numpy as np 
# import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import time
import math
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SimpleTwoDimensional():
    """ Generate two-dimensional points from a simple problem
        in this case: z = cos(x)sin(y)
    """
    def __init__(self):
        self.x_size = np.pi 
        self.y_size = np.pi

    def gather_train_samples(self, n_samples):
        self.training_data = self.uniform_sampling(n_samples)
        self.ntrain_samples = n_samples

    def gather_validation_samples(self, n_samples):
        self.validation_data = self.uniform_sampling(n_samples)

    def simulate_system(self, x, u,robotId ):
        x_next = []
        for i in range(7):
            p.resetJointState(robotId, i, x[i], targetVelocity=x[i + 7])

        p.setJointMotorControlArray(robotId, range(7), controlMode=p.TORQUE_CONTROL, forces=u)
        p.stepSimulation()

        for i in range(7):
            x_next.append(p.getJointStates(robotId, range(7))[i][1])
        x_next = np.array(x_next)
        return x_next

    def gather_test_samples(self, n_samples):
        # self.test_data = self.uniform_sampling(n_samples, self.action_size)

        physicsClient = p.connect(p.GUI)

        p.setGravity(0, 0, -10)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        robotId = p.loadURDF("ILQR/iiwa7.urdf", flags=9, useFixedBase=1)

        robotStartPos = [0, 0, 0]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        p.resetBasePositionAndOrientation(robotId, robotStartPos, robotStartOrientation)

        p.setJointMotorControlArray(robotId, range(7), p.VELOCITY_CONTROL, forces=np.zeros(7))

        x_test = np.array([4.33333333e-01, 3.72222222e-01, 1.00000000e+00, 6.33333333e-01,
                           7.77777778e-01, 5.50000000e-01, 8.77777778e-01, 5.61716944e+00,
                           4.94226108e-01, 7.71834766e+00, 3.36154878e+00, -7.54252866e+00,
                           -4.02101231e+00, -3.41661910e+00, 3.00000000e+00, -1.80000000e+01,
                           2.80000000e+01, 7.00000000e+00, -2.60000000e+01, 1.00000000e+01,
                           1.40000000e+01])
        u_test = x_test[14::]
        yy = self.simulate_system(x_test, u_test, robotId)
        yy = np.reshape(yy,(1,7))
        x_test = np.reshape(x_test,(1,21))
        self.test_data = np.append(x_test, yy, axis=1)

        self.xaxis = x_test[::14]
        self.yaxis = u_test[:]
        p.disconnect(physicsClient)

    def get_batch(self, batch_size):
        """ Sample batch_size (state,action)-pairs from the training data """
        indices = np.random.choice(range(self.ntrain_samples-1),size=batch_size,replace=False)
        return self.training_data[indices]

    def uniform_sampling(self, n_samples):
        """ Uniformly sample (state,action) pairs for model learning 
            - replaces standard dynamical model for testing purposes
        """
        # Sample uniformly from within the domain
        # x = np.random.uniform(-self.x_size, self.x_size, size=n_samples)
        # y = np.random.uniform(self.y_size, self.y_size, size=n_samples)

        # Generate two clusters within the 2D state space
        physicsClient = p.connect(p.GUI)

        p.setGravity(0, 0, -10)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        robotId = p.loadURDF("ILQR/iiwa7.urdf", flags=9, useFixedBase=1)

        robotStartPos = [0, 0, 0]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        p.resetBasePositionAndOrientation(robotId, robotStartPos, robotStartOrientation)

        p.setJointMotorControlArray(robotId, range(7), p.VELOCITY_CONTROL, forces=np.zeros(7))

        N = n_samples
        x = np.zeros([14, N])
        u = np.zeros([7, N])
        x_new = np.zeros([7, N])

        for i in range(N):
            for j in range(7):
                a = random.randint(-180, 180)  # angule range(-pi,pi)
                b = random.uniform(-10, 10)  # velocity range
                c = random.randint(-30, 30)  # torque range (-200,200)
                x[j, i] = math.radians(a / math.pi)
                x[j + 7, i] = b
                u[j, i] = c

        for i in range(N):
            x_new[:, i] = self.simulate_system(x[:, i], u[:, i], robotId)

        x = x.T
        y = u.T
        outputs = x_new.T
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        outputs = outputs.astype(np.float32)
        x_train = np.append(x, y, axis=1)
        data = np.append(x_train, outputs, axis=1)

        p.disconnect(physicsClient)

        return data 

