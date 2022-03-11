import pybullet as p
import pybullet_data
from pybullet_sim_panda.utils import *
import time
from pybullet_sim_panda.dynamics import PandaDynamics
import spatialmath as sm
from eec.eec import EEC
from eec.subfunctions import *
import copy



RATE = 240. # default: 240Hz
REALTIME = 0
DURATION = 10

t = 0.
stepsize = 1/RATE

uid = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

p.setAdditionalSearchPath(pybullet_data.getDataPath()) # for loading plane

p.resetSimulation() #init
p.setRealTimeSimulation(REALTIME)
p.setGravity(0, 0, -9.81) #set gravity

plane_id = p.loadURDF("plane.urdf", useFixedBase=True) # load plane
p.changeDynamics(plane_id,-1,restitution=.95)

panda = PandaDynamics(p, uid) # load robot
panda.setControlMode("torque")









""" Target position and orientation is needed
"""
target_pos = np.array([6.12636866e-01, -3.04817487e-12, 5.54489818e-01], np.float64)
target_ori = np.array([2.77158854, 1.14802956, 0.41420822], np.float64)
# target_v = np.zeros(3, np.float64)
# target_w = np.zeros(3, np.float64)
K_p = 10
K_r = 1
K_d = 1



target_R = sm.base.exp2r(target_ori)

R = sm.base.exp2r(panda.get_ee_pose()[1])
R_past = copy.deepcopy(R)
R_dot = (R-R_past)/stepsize

pos_past = panda.get_ee_pose()[0]
R_e = target_R.T @ R
# if np.trace(R_e) >= 3:
#     eps_e = np.zeros(3, np.float64)
#     R_e = np.eye(3, dtype=np.float64)
# else:
#     eps_e = sm.base.trlog(R_e, check=False, twist=True)
eps_e = trLog(R_e, check=False, twist=True)

eec_panda = EEC(dt=stepsize, theta=np.linalg.norm(eps_e), R_init=R_e)




for i in range(int(DURATION/stepsize)):
    if i%RATE == 0:
        print("Simulation time: {:.3f}".format(t))
    if i%(5*RATE) == 0:
        panda.reset()
        t = 0.
        panda.setControlMode("torque")
        target_torque = [0,0,0,0,0,0,0]
    
    pos, ori = panda.get_ee_pose()
    pos_error = pos - target_pos
    # ori_error = sm.base.trlog(target_R.T @ R, twist=True)
    vel = (pos-pos_past)/stepsize

    R = sm.base.exp2r(ori)
    R_dot = (R-R_past)/stepsize
    R_e = target_R.T @ R

    wb = vee(R.T @ R_dot)
    eec_panda.update(R_e, wb)
    
    V_b = np.concatenate((vel, wb), axis=None)
    d_term = V_b*K_d

    conv_ori = B(eec_panda.get_unit_vector()*eec_panda._theta) @ ((R_e.T @ sm.base.exp2r(eec_panda._eec)).T)
    Convert = np.concatenate((np.concatenate((R, np.zeros((3,3), np.float64)), axis=1), np.concatenate((np.zeros((3,3), np.float64), conv_ori), axis=1)), axis=0)
    p_term = np.concatenate((pos_error*K_p, eec_panda._eec*K_r), axis=None)
    p_term = Convert.T @ p_term
    
    
    Fb = -d_term - p_term
    print(Fb)
    print(panda.get_body_jacobian())
    # print(eec_panda._eec)




    target_torque = [0,0,0,0,0,0,0]
    panda.setTargetTorques(target_torque)
    # print(panda.get_ee_pose())


    t += stepsize
    p.stepSimulation()
    time.sleep(stepsize)

    R_past = copy.deepcopy(R)
    pos_past = pos