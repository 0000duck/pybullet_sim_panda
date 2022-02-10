from eec.eec import EEC
import numpy as np
import spatialmath as sm
import matplotlib.pyplot as plt


def Omega(t):
    pi = np.pi
    return np.array([pi*np.cos(pi*t), -2*pi*np.sin(pi*t), 3]).astype(np.float64)



## Initialization
'''
k: number of turns to be stored
theta*u
k_0
theta_b12
theta_b23
dt
eec
'''

t = 0
R = np.eye(3).astype(np.float64)
# w = (4*np.pi/np.sqrt(3)) * np.array([1., 1., 1.]).astype(np.float64)
w = Omega(t)

theta_b12 = np.pi/2
theta_b23 = 0.015
gain = 10
dt = 0.001

k = 0
theta = 0


eec = EEC(dt=dt, theta=theta)


# Variables for plots
t_space = []
eec_x_space = []
eec_y_space = []
eec_z_space = []
ec_x_space = []
ec_y_space = []
ec_z_space = []
eec_space = []
ec_space = []
error_x_space = []
error_y_space = []
error_z_space = []




## Loop
while t <= 10:
    print("Time: ", t)
    print("Theta is:", eec._theta)
    print("k is: ", eec._k)
    print("EEC:")
    print(eec._eec)
    print("\n")

    
    eec.update(R=R, w=w)


    # Data for plots
    t_space.append(t)
    # print(R)
    try:
        ec = sm.base.trlog(R, check=False, twist=True)
    except:
        ec = np.zeros(3).astype(np.float64)
    ec_x_space.append(ec[0])
    ec_y_space.append(ec[1])
    ec_z_space.append(ec[2])
    ec_space.append(np.linalg.norm(ec))
    eec_x_space.append(eec._eec[0])
    eec_y_space.append(eec._eec[1])
    eec_z_space.append(eec._eec[2])
    eec_space.append(np.linalg.norm(eec._eec))
    error_x_space.append(eec._error[0])
    error_y_space.append(eec._error[1])
    error_z_space.append(eec._error[2])



    # Time update
    t += dt
    # Rotation matrix update
    R = R @ sm.base.exp2r(dt*w)
    # Angular velocity update
    w = Omega(t)



plt.subplot(5,1,1)
plt.plot(t_space, ec_x_space, color='b', label="ec_x")
plt.plot(t_space, eec_x_space, '--', color='r', label="eec_x")
plt.legend()
plt.subplot(5,1,2)
plt.plot(t_space, ec_y_space, color='b', label="ec_y")
plt.plot(t_space, eec_y_space, '--', color='r', label="eec_y")
plt.legend()
plt.subplot(5,1,3)
plt.plot(t_space, ec_z_space, color='b', label="ec_z")
plt.plot(t_space, eec_z_space, '--', color='r', label="eec_z")
plt.legend()
plt.subplot(5,1,4)
plt.plot(t_space, ec_space, color='b', label="theta")
plt.plot(t_space, eec_space, '--', color='r', label="theta_bar")
plt.legend()
plt.subplot(5,1,5)
plt.plot(t_space, error_x_space, color='b', label="error_x")
plt.plot(t_space, error_y_space, '--', color='r', label="error_y")
plt.plot(t_space, error_z_space, '-.', color='g', label="error_z")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Angle(rad)")
plt.show()