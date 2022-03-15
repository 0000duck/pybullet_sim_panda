from eec.subfunctions import *

import spatialmath as sm
import numpy as np
import copy



class EEC:
    def __init__(self, dt, theta, k=0, R_init=np.eye(3).astype(np.float64), theta_b12=np.pi/2, theta_b23=0.015, gain=10, buffer_size=3):
        self._R = R_init
        self._k = k
        self._theta = theta
        self._eec = (2*k*np.pi + theta) * trLog(R_init, twist=True)
        
        self._R_bar = sm.base.exp2r(self._eec)
        self._error = trLog(self._R_bar.T @ R_init, check=False, twist=True)

        self._BUFFER_SIZE = buffer_size
        self._buffer_eec = []
        self._buffer_unit = []

        if self._theta != 0:
            for i in range(self._BUFFER_SIZE):
                self._buffer_eec.append(copy.deepcopy(self._eec))
                self._buffer_unit.append(self._buffer_eec[i] / np.linalg.norm(self._buffer_eec[i]))
        else:
            for i in range(self._BUFFER_SIZE):
                self._buffer_eec.append(copy.deepcopy(self._eec))
                self._buffer_unit.append(np.array([0,0,1], np.float64))

        
        self._dt = dt
        self._theta_b12 = theta_b12
        self._theta_b23 = theta_b23
        self._gain = gain


    def get_unit_vector(self):
        return self._buffer_unit[0]


    def update(self, R, w):
        self._R_bar = sm.base.exp2r(self._eec)
        self._error = trLog(self._R_bar.T @ R, check=False, twist=True)
        
        w_bar = self._gain*self._error + self._R_bar.T @ R @ w

        eec_norm = np.linalg.norm(self._eec)
        self._k, self._theta = eec_norm//(2*np.pi), eec_norm%(2*np.pi)
        if self._theta > np.pi:
            self._theta -= 2*np.pi
            self._k += 1
        
        if abs(self._theta) > self._theta_b12:
            # print("Algorithm 1")
            theta_bar = 2*self._k*np.pi + self._theta
            unit = self._eec / eec_norm
            U = np.array([[theta_bar, 0, 0, unit[0]], [0, theta_bar, 0, unit[1]], [0, 0, theta_bar, unit[2]]], np.float64)
            A = np.array([[self._theta, 0, 0, unit[0]], [0, self._theta, 0, unit[1]], [0, 0, self._theta, unit[2]], [unit[0], unit[1], unit[2], 0]], np.float64)
            temp = B(self._theta*unit) @ w_bar
            b = np.array([temp[0], temp[1], temp[2], 0])
            eec_dot = U @ np.linalg.inv(A) @ b
            self._eec += self._dt * eec_dot

            # Buffer update
            for i in range(self._BUFFER_SIZE-1, 0, -1):
                self._buffer_eec[i] = copy.deepcopy(self._buffer_eec[i-1])
                self._buffer_unit[i] = copy.deepcopy(self._buffer_unit[i-1])
            self._buffer_eec[0] = copy.deepcopy(self._eec)
            self._buffer_unit[0] = self._eec / np.linalg.norm(self._eec)

        elif abs(self._theta) > self._theta_b23 or self._k == 0 or self._theta == np.pi:
            # print("Algorithm 2")
            temp_vec = trLog(self._R_bar @ sm.base.exp2r(self._dt*w_bar), twist=True)
            theta_next = np.linalg.norm(temp_vec)
            if theta_next == 0:
                u_next = np.array([0,0,1], np.float64)
            else:
                u_next = temp_vec / theta_next

            eec_1 = (2*self._k*np.pi + theta_next)*u_next
            eec_2 = (-2*self._k*np.pi + theta_next)*u_next
            if np.linalg.norm(self._eec-eec_1) <= np.linalg.norm(self._eec-eec_2):
                self._eec = eec_1
            else:
                self._eec = eec_2

            # Buffer update
            for i in range(self._BUFFER_SIZE-1, 0, -1):
                self._buffer_eec[i] = copy.deepcopy(self._buffer_eec[i-1])
                self._buffer_unit[i] = copy.deepcopy(self._buffer_unit[i-1])
            self._buffer_eec[0] = copy.deepcopy(self._eec)
            self._buffer_unit[0] = u_next

        else:
            # print("Algorithm 3")
            R_bar_next = self._R_bar @ sm.base.exp2r(self._dt*w_bar)
            if np.trace(R_bar_next) >= 3:
                theta_next = 0
                u_next = np.array([0., 0., 1.]).astype(np.float64)
            else:
                temp_vec = trLog(R_bar_next, twist=True)
                theta_next = np.linalg.norm(temp_vec)
                u_next = temp_vec / theta_next

            S1 = self._buffer_unit[:]
            S1.insert(0, u_next)
            S2 = self._buffer_unit[:]
            S2.insert(0, -u_next)
            S1_mean = np.zeros(3).astype(np.float64)
            S2_mean = np.zeros(3).astype(np.float64)
            for i in range(self._BUFFER_SIZE+1):
                S1_mean += S1[i]
            S1_mean /= (self._BUFFER_SIZE+1)
            for i in range(self._BUFFER_SIZE+1):
                S2_mean += S2[i]
            S2_mean /= (self._BUFFER_SIZE+1)
            
            dist = 0
            idx1 = 0
            for i, u in enumerate(S1):
                d = np.linalg.norm(u-S1_mean)
                if dist < d:
                    idx1 = i
                    dist = d
            S1.pop(idx1)

            dist = 0
            idx2 = 0
            for i, u in enumerate(S2):
                d = np.linalg.norm(u-S2_mean)
                if dist < d:
                    idx2 = i
                    dist = d
            S2.pop(idx2)
            
            idx1_bool = bool(idx1)
            idx2_bool = bool(idx2)

            if idx1_bool^idx2_bool == False:
                u_star = np.mean(self._buffer_unit, axis=0)
                u_star /= np.linalg.norm(u_star)
                eec_1 = (2*self._k*np.pi + theta_next) * u_star
                eec_2 = (2*self._k*np.pi - theta_next) * u_star
                R_bar_1 = sm.base.exp2r(eec_1)
                R_bar_2 = sm.base.exp2r(eec_2)
                if np.linalg.norm(R_bar_1-R_bar_next, 'fro') <= np.linalg.norm(R_bar_2-R_bar_next, 'fro'):
                    self._eec = eec_1
                else:
                    self._eec = eec_2
            else:
                if idx1_bool == True:
                    u_star = np.mean(S1 ,axis=0)
                    u_star /= np.linalg.norm(u_star)
                    self._eec = (2*self._k*np.pi + theta_next) * u_star
                else:
                    u_star = np.mean(S2 ,axis=0)
                    u_star /= np.linalg.norm(u_star)
                    self._eec = (2*self._k*np.pi - theta_next) * u_star
        
            # Buffer update
            for i in range(self._BUFFER_SIZE-1, 0, -1):
                self._buffer_eec[i] = copy.deepcopy(self._buffer_eec[i-1])
                self._buffer_unit[i] = copy.deepcopy(self._buffer_unit[i-1])
            self._buffer_eec[0] = copy.deepcopy(self._eec)
            self._buffer_unit[0] = u_star