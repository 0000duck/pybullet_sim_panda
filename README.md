# pybullet_sim_panda

Pybullet simulation environment for Franka Emika Panda

### Dependency
pybullet, numpy, spatial_math_mini

### Simple example (please check sim_example.ipynb)
```python
uid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # for loading plane

# Load
plane_id = p.loadURDF("plane.urdf") # load plane
panda = PandaKinematics(p, uid) # load robot

## kinematics usage
panda.set_arm_positions([0,0,0,-1,2,1,2])
states = panda.get_states()
joints = panda.get_arm_positions()
panda.set_home_positions()
panda.open() #gripper
pos, ori = panda.get_link_pose(3)

## example(jacobian-pseudoinv)
panda.set_home_positions()
pos_curr, _ = panda.get_ee_pose()
pos_goal = pos_curr + 0.1

# show current/goal position
clear()
view_point(pos_curr) # show the frame for debugging purpose
view_point(pos_goal)

# control-like kinematics simulation
for i in range(10):
    jac = panda.get_space_jacobian()[3:]
    q_delta = np.linalg.pinv(jac) @ (pos_goal - pos_curr)
    q_new = panda.get_arm_positions() + q_delta*0.1
    panda.set_arm_positions(q_new)
    time.sleep(0.5)
```
