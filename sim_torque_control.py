import pybullet as p
import pybullet_data
from pybullet_sim_panda.utils import *
import time
from pybullet_sim_panda.dynamics import PandaDynamics
import spatialmath as sm
from eec.eec import EEC
from eec.subfunctions import *
import copy
import matplotlib.pyplot as plt



def reverseTwist(twist):
    a, b = twist[0:3], twist[3:]
    return np.concatenate([b, a], axis=None)



RATE = 240. # default: 240Hz
REALTIME = 0
DURATION = 4000
STEPSIZE = 1/RATE

t = 0.

uid = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

p.setAdditionalSearchPath(pybullet_data.getDataPath()) # for loading plane

p.resetSimulation() #init
p.setRealTimeSimulation(REALTIME)
p.setGravity(0, 0, 0) #set gravity

plane_id = p.loadURDF("plane.urdf", useFixedBase=True) # load plane
p.changeDynamics(plane_id,-1,restitution=.95)

panda = PandaDynamics(p, uid) # load robot
# panda.set_arm_positions([0,0,0,-np.pi/2,0,np.pi/2,np.pi/1.5])
panda.setControlMode("torque")

""" To make the damping terms zero
"""
# for link_idx in panda._arm_joints:
#     p.changeDynamics(panda._robot, link_idx, jointDamping=0)



""" Target position and orientation is needed
"""
target_pos = np.array([6.12636866e-01, -3.04817487e-12, 5.54489818e-01], np.float64)
target_ori = np.array([2.77158854, 1.14802956, 0.41420822], np.float64)
target_R = sm.base.exp2r(target_ori)
K_p = 12 # propotional(positional) gain
K_r = 5 # propotional(rotational) gain
K_d = 0.5 # damping gain


R = sm.base.exp2r(panda.get_ee_pose(exp_flag=True)[1])
R_past = copy.deepcopy(R)
R_dot = (R-R_past)/STEPSIZE

pos_past = panda.get_ee_pose(exp_flag=True)[0]
R_e = target_R.T @ R
R_e_past = copy.deepcopy(R_e)

eec_panda = EEC(dt=STEPSIZE, R_init=R_e, k=3)
# For checking the initial eec
theta_bar = np.linalg.norm(eec_panda._eec)
print(eec_panda._eec)


time_list = []
theta_bar_list = []
pos_error_list = []
rot_error_list = []






for i in range(int(DURATION/STEPSIZE)):
    if i%RATE == 0:
        print("Simulation time: {:.3f}".format(t))
    

    pos, ori = panda.get_ee_pose(exp_flag=True)
    # print(pos)
    # print(ori)
    pos_error = pos - target_pos
    vel = (pos-pos_past)/STEPSIZE

    R = sm.base.exp2r(ori)
    R_dot = (R-R_past)/STEPSIZE
    R_e = target_R.T @ R
    R_e_dot = (R_e-R_e_past)/STEPSIZE

    vel_b = R.T @ vel
    w_b = vee(R_e.T @ R_e_dot)
    # w_b = vee(R.T @ R_dot)
    eec_panda.update(R_e, w_b)

    # print(R)
    # print(R_dot)
    # print(R.T @ R_dot)
    # print("\n")
    
    V_b = np.concatenate((vel_b, w_b), axis=None)
    d_term = V_b*K_d

    conv_ori = B(eec_panda.get_unit_vector()*eec_panda._theta) @ ((R_e.T @ sm.base.exp2r(eec_panda._eec)).T)
    Convert = np.concatenate((np.concatenate((R, np.zeros((3,3), np.float64)), axis=1),
                              np.concatenate((np.zeros((3,3), np.float64), conv_ori), axis=1)), axis=0)
    p_term = np.concatenate((pos_error*K_p, eec_panda._eec*K_r), axis=None)
    p_term = Convert.T @ p_term
    
    # (pos, ori) to (ori, pos)
    p_term = reverseTwist(p_term)
    d_term = reverseTwist(d_term)

    Fb = -d_term - p_term
    Jb = panda.get_body_jacobian()

    tau = Jb.T @ Fb
    # print("==========Torque==========")
    # # print(tau)
    # print(panda._target_torque)
    # print("============EEC============")
    # print(eec_panda._eec)


    tau_grav = panda.inverseDynamics(panda.get_states("all")["position"], [0.]*9, [0.]*9)[:-2]

    target_torque = tau + tau_grav
    # target_torque = [0] * 7
    panda.setTargetTorques(target_torque, saturate=True)

    theta_bar = np.linalg.norm(eec_panda._eec)
    # if theta_bar > np.pi:
    #   print(theta_bar)
    # print(eec_panda._k)

    t += STEPSIZE
    p.stepSimulation()
    time.sleep(STEPSIZE)

    R_past = copy.deepcopy(R)
    pos_past = pos
    R_e_past = copy.deepcopy(R_e)





    """ For data plotting
    """
    time_list.append(t)
    theta_bar_list.append(theta_bar)
    pos_error_list.append(np.linalg.norm(pos_error))

plt.figure(1)
plt.plot(time_list, theta_bar_list)
plt.xlabel("Time(s)")
plt.ylabel("Angle(rad)")
plt.figure(2)
plt.plot(time_list, pos_error_list)
plt.xlabel("Time(s)")
plt.ylabel("Positional error(m)")
plt.show()
