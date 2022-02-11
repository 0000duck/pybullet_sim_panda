import pybullet as p
import pybullet_data
from pybullet_sim_panda.utils import *
import time
from pybullet_sim_panda.dynamics import PandaDynamics



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
# p.setTimeStep(1/RATE)
p.setRealTimeSimulation(REALTIME)
p.setGravity(0, 0, -9.81) #set gravity

plane_id = p.loadURDF("plane.urdf", useFixedBase=True) # load plane
p.changeDynamics(plane_id,-1,restitution=.95)

panda = PandaDynamics(p, uid) # load robot
# panda.setControlMode("torque")
panda.setControlMode("position")

for i in range(int(DURATION/stepsize)):
    if i%RATE == 0:
        print("Simulation time: {:.3f}".format(t))
    if i%(5*RATE) == 0:
        panda.reset()
        t = 0.
        # panda.setControlMode("torque")
        panda.setControlMode("position")
        # target_torque = [0,0,0,0,0,0,0]
    
    # target_torque = [1,0,0,0,0,0,0]
    # panda.setTargetTorques(target_torque)
    panda.setTargetPositions(panda._joint_lower_limits)
    print(panda.get_ee_pose()[1])


    t += stepsize
    p.stepSimulation()
    time.sleep(stepsize)