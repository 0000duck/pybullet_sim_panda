import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import copy



def B(eps):
    norm_eps = np.linalg.norm(eps)
    skew_eps = np.array([[0, -eps[2], eps[1]], [eps[2], 0, -eps[0]], [-eps[1], eps[0], 0]]).astype(np.float64)
    alpha = (norm_eps/2)/np.tan(norm_eps/2)
    return np.eye(3) + skew_eps/2 + ((1-alpha)/(norm_eps**2)) * (skew_eps@skew_eps)

def Omega(t):
    return np.array([0.000002*np.sin(t), -0.000001*np.sin(t), 0.000002*np.cos(t)]).astype(np.float64)



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

k = 1
theta = 0
eec = (2*k*np.pi + theta) * sm.base.trlog(R, twist=True)
if np.linalg.norm(eec) == 0:
    eec = (2*k*np.pi + theta) * np.array([0,0,1.]).astype(np.float64)

BUFFER_SIZE = 3
buffer_eec = []
buffer_unit = []
for i in range(BUFFER_SIZE):
    buffer_eec.append(copy.deepcopy(eec))
    buffer_unit.append(buffer_eec[i] / np.linalg.norm(buffer_eec[i]))


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
    print("Theta is:", theta)
    print("k is: ", k)
    print("EEC:")
    print(eec)
    print("\n")

    
    # Update measurements
    '''
    R_bar
    eps_tilde
    Omg_bar
    eec
    '''
    R_bar = sm.base.exp2r(eec)
    # print(R_bar.T@R)
    try:
        error = sm.base.trlog(R_bar.T @ R, check=False, twist=True)
    except:
        error = np.zeros(3).astype(np.float64)



    w_bar = gain*error + R_bar.T @ R @ w
    
    norm_eec = np.linalg.norm(eec)
    k, theta = norm_eec // (2*np.pi), norm_eec % (2*np.pi)
    if theta > np.pi:
        theta -= 2*np.pi
        k += 1

    # Algorithm 1
    if abs(theta) > theta_b12:
        print("Algorithm 1")
        theta_bar = 2*k*np.pi + theta
        unit = eec / norm_eec
        U = np.array([[theta_bar, 0, 0, unit[0]], [0, theta_bar, 0, unit[1]], [0, 0, theta_bar, unit[2]]]).astype(np.float64)
        A = np.array([[theta, 0, 0, unit[0]], [0, theta, 0, unit[1]], [0, 0, theta, unit[2]], [unit[0], unit[1], unit[2], 0]]).astype(np.float64)
        temp = B(theta*unit) @ w_bar
        b = np.array([temp[0], temp[1], temp[2], 0])
        eec_dot = U @ np.linalg.inv(A) @ b

        eec += dt * eec_dot
    
    # Algorithm 2
    elif abs(theta) > theta_b23 or k == 0 or theta == np.pi:
        print("Algorithm 2")
        temp_vec = sm.base.trlog(R_bar @ sm.base.exp2r(dt*w_bar), twist=True)
        theta_next = np.linalg.norm(temp_vec)
        u_next = temp_vec / theta_next

        eec_1 = (2*k*np.pi + theta_next)*u_next
        eec_2 = (-2*k*np.pi + theta_next)*u_next

        if np.linalg.norm(eec-eec_1) <= np.linalg.norm(eec-eec_2):
            eec = eec_1
        else:
            eec = eec_2
    
    # Algorithm 3
    else:
        print("Algorithm 3")
        R_bar_next = R_bar @ sm.base.exp2r(dt*w_bar)
        if np.trace(R_bar_next) >= 3:
            theta_next = 0
            u_next = np.array([0., 0., 1.]).astype(np.float64)
        else:
            temp_vec = sm.base.trlog(R_bar_next, twist=True)
            theta_next = np.linalg.norm(temp_vec)
            u_next = temp_vec / theta_next

        S1 = buffer_unit[:]
        S1.insert(0, u_next)
        S2 = buffer_unit[:]
        S2.insert(0, -u_next)
        S1_mean = np.zeros(3).astype(np.float64)
        S2_mean = np.zeros(3).astype(np.float64)
        for i in range(BUFFER_SIZE+1):
            S1_mean += S1[i]
        S1_mean /= (BUFFER_SIZE+1)
        for i in range(BUFFER_SIZE+1):
            S2_mean += S2[i]
        S2_mean /= (BUFFER_SIZE+1)
        
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
            u_star = np.mean(buffer_unit, axis=0)
            u_star /= np.linalg.norm(u_star)
            eec_1 = (2*k*np.pi + theta_next) * u_star
            eec_2 = (2*k*np.pi - theta_next) * u_star
            R_bar_1 = sm.base.exp2r(eec_1)
            R_bar_2 = sm.base.exp2r(eec_2)
            if np.linalg.norm(R_bar_1-R_bar_next, 'fro') <= np.linalg.norm(R_bar_2-R_bar_next, 'fro'):
                eec = eec_1
            else:
                eec = eec_2
        else:
            if idx1_bool == True:
                u_star = np.mean(S1 ,axis=0)
                u_star /= np.linalg.norm(u_star)
                eec = (2*k*np.pi + theta_next) * u_star
            else:
                u_star = np.mean(S2 ,axis=0)
                u_star /= np.linalg.norm(u_star)
                eec = (2*k*np.pi - theta_next) * u_star



    # Data for plots
    t_space.append(t)
    # print(R)
    try:
        ec = sm.base.trlog(R, check=False, twist=True)
    except:
        ec = np.array([0,0,1]).astype(np.float64)
    ec_norm = np.linalg.norm(ec)
    ec_x_space.append(ec[0]/ec_norm)
    ec_y_space.append(ec[1]/ec_norm)
    ec_z_space.append(ec[2]/ec_norm)
    ec_space.append(np.linalg.norm(ec))
    eec_norm = np.linalg.norm(eec)
    eec_x_space.append(eec[0]/eec_norm)
    eec_y_space.append(eec[1]/eec_norm)
    eec_z_space.append(eec[2]/eec_norm)
    eec_space.append(np.linalg.norm(eec))
    error_x_space.append(error[0])
    error_y_space.append(error[1])
    error_z_space.append(error[2])


    for i in range(BUFFER_SIZE-1, 0, -1):
        buffer_eec[i] = copy.deepcopy(buffer_eec[i-1])
        buffer_unit[i] = copy.deepcopy(buffer_unit[i-1])
    buffer_eec[0] = copy.deepcopy(eec)
    buffer_unit[0] = eec / np.linalg.norm(eec)



    # Time update
    t += dt
    # Rotation matrix update
    R = R @ sm.base.exp2r(dt*w)
    # Angular velocity update
    w = Omega(t)

    


    




# print(eec_space)

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