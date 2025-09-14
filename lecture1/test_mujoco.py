
import mujoco
import mujoco.viewer
import numpy as np
import os
import time 

# Path to your MJCF file
xml_path = os.path.join(os.path.dirname(__file__), "pendulum.xml")

# Load model and data
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Set initial condition from the keyframe
mujoco.mj_resetData(model, data)
data.qpos[0] = 0.5236   # 30 degrees
data.qvel[0] = 0.0      # zero angular velocity
print("data.qpos[0]",data.qpos[0])
mujoco.mj_forward(model, data)  # recompute positions, forces
print("Initial qpos:", float(data.qpos[0]))        
    

# Run simulation with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        start = time.time()

        # empty control
        data.ctrl[0] = 0.0

        # Advance physics
        mujoco.mj_step(model, data)
        step += 1

        print("qpos in sim:", float(data.qpos[0]))

        # Sync viewer at realtime rate
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
