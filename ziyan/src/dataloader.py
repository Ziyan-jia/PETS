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

class Sinewave():
    """ Generate points from a given sine wave """
    def __init__(self, N, seed):
        self.xmin = -2*np.pi
        self.xmax = 2*np.pi
        np.random.seed(seed) 
        self.name = "sinewave"

        # Divide interval in f parts and generate samples from the first and last
        f = 1
        xl = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin 
        xr = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin + (f-1) * (self.xmax - self.xmin) / f
        self.x = np.concatenate((xl, xr), axis=0)
        scale = 0.0225 * np.abs(np.sin(1.5 * self.x + np.pi / 8))
        noise = np.random.normal(0, scale=scale, size=len(self.x))
        self.y = np.sin(self.x) + noise

        # Standardize data
        # self.input_mean = np.mean(self.x)
        # self.input_std = np.std(self.x)
        # self.x = (self.x-self.input_mean)/self.input_std

    def get_samples(self, N):
        """ Get N samples from the generated dataset """
        indices = np.random.choice(np.arange(len(self.x)), size=N)
        x = self.x[indices]
        y = self.y[indices]
        return x, y
    
    def get_test_samples(self, N):
        a = 0.5
        delta = self.xmax - self.xmin 
        x = np.linspace(self.xmin-a*delta, self.xmax+a*delta, num=N)
        y = np.sin(x)
        # Standardize data
        # x = (x-self.input_mean)/self.input_std 
        return x, y

class Simplecurve():
    """ Generate points from a simple curve
        in this case, y=x**3
    """
    def __init__(self, N, seed):
        self.xmax = -2
        self.xmin = 2
        np.random.seed(seed)
        self.name = "simplecurve"
        
        # Divide interval in f parts and generate samples from the first and last
        f = 1
        xl = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin 
        xr = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin + (f-1) * (self.xmax - self.xmin) / f
        self.x = np.concatenate((xl, xr), axis=0)
        noise = np.random.normal(0, 0.05*self.xmax**2, size=len(self.x))
        self.y = self.x**2 + noise 
        # Standardize data
        # self.input_mean = np.mean(self.x)
        # self.input_std = np.std(self.x)
        # self.x = (self.x-self.input_mean)/self.input_std

    def get_samples(self, N):
        """ Get N samples from the generated dataset """
        indices = np.random.choice(np.arange(len(self.x)), size=N)
        x = self.x[indices]
        y = self.y[indices]
        return x, y

    def get_test_samples(self, N):
        a = 0.5
        x = np.linspace(self.xmin-a*(self.xmax-self.xmin), self.xmax+a*(self.xmax-self.xmin), num=N)
        y = x**2
        # Standardize data
        # x = (x-self.input_mean)/self.input_std 
        return x, y

class SimpleTwoDimensional():
    """ Generate two-dimensional points from a simple problem
        in this case: z = cos(x)sin(y)
    """
    def __init__(self, seed):
        self.x_size = np.pi 
        self.y_size = np.pi
        np.random.seed(seed)

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
            x_next.append(p.getJointStates(robotId, range(7))[i][0])
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

        N = int(n_samples)
        x = np.zeros([14, N])
        u = np.zeros([7, N])
        x_new = np.zeros([14, N])

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

        xx = x.T
        u = u.T
        x_train = np.append(xx, u, axis=1)
        y_train = x_new.T
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        x = torch.from_numpy(x_train)
        y = torch.from_numpy(y_train)
        self.test_data = np.zeros((n_samples, 35))
        self.test_data = np.append(x, y, axis=1)
        
        self.xaxis = xx[:]
        self.yaxis = u[:]
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
        x_new = np.zeros([14, N])

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
        u = u.T
        x_train = np.append(x, u, axis=1)
        y_train = x_new.T
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        x = torch.from_numpy(x_train)
        y = torch.from_numpy(y_train)
        data = np.append(x, y, axis=1)


        self.xaxis = x[:]
        self.yaxis = y[:]
        p.disconnect(physicsClient)


        return data 

    def system(self, x, y):
        """ Define system evolution """
        output = np.sin(x) * np.sin(y)
        return output