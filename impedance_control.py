import pybullet as p
import pybullet_data
import time
from pybullet_sim_panda.dynamics import PandaDynamics
import spatialmath as sm
import numpy as np


## utility functions
def qtn_to_mat(qtn):
    return np.array(p.getMatrixFromQuaternion(qtn)).reshape(3,3)

def mat_to_exp_coord(R):
    """[summary]
    Convert rotation matrix to unit quaternion
    (logarithm of SO3)
    Args:
        R (3x3 np.ndarray): Rotation matrix.
    Returns:
        axis [size 3 np.ndarray]: Unit axis.
        angle (float): rotation angle
    """
    if np.allclose(R, np.eye(3)):
        # no rotation
        axis = np.array([1., 0., 0.])
        angle = 0.
    elif np.trace(R) == -1:
        # angle is 180 degrees
        angle = np.pi
        if R[0,0] != -1:
            axis = 1/np.sqrt(2*(1+R[0,0])) * np.array([1+R[0,0], R[1,0], R[2,0]])
        elif R[1,1] != -1:
            axis = 1/np.sqrt(2*(1+R[1,1])) * np.array([R[0,1], 1+R[1,1], R[2,1]])
        else:
            axis = 1/np.sqrt(2*(1+R[2,2])) * np.array([R[0,2], R[1,2], 1+R[2,2]])
    else:
        angle = np.arccos(1/2*(np.trace(R)-1))
        axis = 1/(2*np.sin(angle))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return axis*angle

# --pseudocode for cartesian impedance control--


RATE = 240. # default: 240Hz
REALTIME = 0
DURATION = 60
STEPSIZE = 1/RATE

# t = 0.

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

# configuration
stiffness = np.diag([*[500]*3, *[10]*3])
damping = np.diag([*[2.*np.sqrt(500)]*3, *[2.*np.sqrt(1)]*3])

# desired pose
pos_d, ori = panda.get_ee_pose(True)
orn_d_mat = sm.base.exp2r(ori)

# control loop
while True:
    # impedance control
    pos_curr, ori_curr = panda.get_ee_pose()
    orn_curr_mat = sm.base.exp2r(ori)
    joint_velocities = panda.get_joint_velocities()

    pos_err = pos_curr - pos_d
    orn_err = orn_curr_mat @ mat_to_exp_coord(orn_d_mat.T @ orn_curr_mat)
    error = np.array([*pos_err, *orn_err]) # screw motion error
    jac = panda.get_ee_jacobian()
    tau_impedance = jac.T @ (-stiffness @ error[:,None] - damping @ (jac @ joint_velocities[:,None]))
    tau_impedance = tau_impedance.flatten() #1d vector form

    # nonlinear dynamics feedforward term
    joint_angles = panda.get_joint_angles(idx_type="movable").tolist()
    joint_velocities = panda.get_joint_velocities(idx_type="movable").tolist()
    joint_accelerations = np.zeros_like(joint_angles).tolist()
    tau_coriolis_gravity = panda.inverse_dynamics(
        joint_positions=joint_angles,
        joint_velocities=joint_velocities,
        joint_accelerations=joint_accelerations
    )
    tau_coriolis_gravity = tau_coriolis_gravity[:-2] # remove finger joint torque

    tau = tau_impedance + tau_coriolis_gravity
    panda.control_joint_torques(tau)
    
    p.stepSimulation()
    time.sleep(STEPSIZE)