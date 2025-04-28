import numpy as np
from scipy import linalg
from scipy import sparse
import osqp
import matplotlib.pyplot as plt
import random

import ai4rgym
from ai4rgym.envs.road import Road
import gymnasium as gym
from utils import ensure_dir, ensure_dirs, eval_model

from rti import (
    RTI,
    bicycle_model_parameters,
    road_elements_list,
    numerical_integration_parameters,
    termination_parameters,
    initial_state_bounds,
    observation_parameters,
)



def create_env(road_elements_list):
    env = gym.make(
        "ai4rgym/autonomous_driving_env",
        render_mode=None,
        bicycle_model_parameters=bicycle_model_parameters,
        road_elements_list=road_elements_list,
        numerical_integration_parameters=numerical_integration_parameters,
        termination_parameters=termination_parameters,
        initial_state_bounds=initial_state_bounds,
        observation_parameters=observation_parameters,
    )

    # Set the integration method and time step
    Ts_sim = 0.05
    integration_method = "rk4"
    env.unwrapped.set_integration_method(integration_method)
    env.unwrapped.set_integration_Ts(Ts_sim)

    # Set the road condition
    env.unwrapped.set_road_condition(road_condition="wet")

    return env



def main():

    py_init_min = 0.0
    v_init_min_in_kmh = 80.0

    print("Starting RTI-MPC!")
    my_env = create_env(road_elements_list = road_elements_list)


    sampling_time = numerical_integration_parameters["Ts"]
    horizon_N = 10

    # Objective matrices
    Q_input = sparse.diags([0., .0, 0., 10.], format="csc", dtype=np.float32) # (progress, orthogonal distance, heading angle, velocity)
    R_input = sparse.diags([.1, .8], format="csc", dtype=np.float32) # F, steering angle

    Q_ref_input = np.diag([0., .0, 0., 10.]) # (progress, orthogonal distance, heading angle, velocity)
    R_ref_input = np.diag([.1, .8])

    rti = RTI(my_env, sampling_time, horizon_N, Q_input, R_input, Q_ref_input, R_ref_input, 0.02, 0.007)

    observation, info_dict = my_env.reset()

    t = 0
    total_sim_time = 500.0

    des_speed_ms = 80 * (1.0/3.6)
    p0 = 0.
    d0 = py_init_min
    mu0 = 0.
    v0 = v_init_min_in_kmh * (1.0/3.6)

    trajectory = {"time":[], 
                "state":[], 
                "action":[],
                "obs": [],
                "inf":[]
                }
    
    positions = []
    deviations = []
    heading_angles = []
    velocities = []
    ground_truth_px = []
    ground_truth_py = []

    actions = []


    s0 = np.array([[p0],[d0],[mu0],[v0]])
    a0 = np.array([[0.0], [0.0]]) # F, steering angle
    
    A_N, B_N, g_N = rti.init_construct_ABg(s0, a0)

    s_ref, a_ref = rti.create_reference(t, des_speed_ms, v0, s0, True)

    P_osqp = rti.mpc_setup()
    q_osqp, A_osqp, l_osqp, u_osqp = rti.mpc_update(A_N, B_N, g_N, s_ref, a_ref, s0)

    # Initialize an OSQP object
    osqp_obj = osqp.OSQP()

    # Setup the parameters of the optimization program
    osqp_obj.setup(
        P=P_osqp,
        q=q_osqp,
        A=A_osqp,
        l=l_osqp,
        u=u_osqp,
        verbose=False,
    )

    # Solve the optimization program
    osqp_result = osqp_obj.solve()

    # Extract the status string
    osqp_status_string = osqp_result.info.status

    # Display the status if it is anything other than success
    if (osqp_status_string != "solved"):
        raise ValueError("OSQP did not solve the problem, returned status = " + osqp_status_string)

    # Extract the optimal solution
    x_optimal = osqp_result.x

    s_opt, a_opt = rti.extract_values(x_optimal)


    trajectory["time"].append(t)
    trajectory["action"].append(a_opt[0])
    trajectory["state"].append(s0)
    trajectory["obs"].append(observation)
    trajectory["inf"].append(info_dict)

    a0 = a_opt[0].reshape(2,)


    positions.append(s0[0,0])
    deviations.append(s0[1,0])
    heading_angles.append(s0[2,0])
    velocities.append(s0[3,0] * 3.6)
    actions.append(a0)
    ground_truth_px.append(info_dict["ground_truth_px"][0])
    ground_truth_py.append(info_dict["ground_truth_py"][0])

    obs, _, terminated, truncated, inf = my_env.step(a0)


    t+= sampling_time

    # while t < total_sim_time:
    print("Looping...")
    while not(terminated) and not(truncated) and (t < total_sim_time):

        s_obs = rti.get_sate_observation(obs, inf)

        s_list = [s_obs] + s_opt[2:]

        a_list = a_opt[1:] + [a_opt[-1]]
        
        A_N, B_N, g_N = rti.dynamics_setup(s_list, a_list)

        s_ref, a_ref = rti.create_reference(t, des_speed_ms, des_speed_ms, s_obs)

        osqp_obj = osqp.OSQP() 

        q_update, A_update, l_update, u_update = rti.mpc_update(A_N, B_N, g_N, s_ref, a_ref, s_obs)
        osqp_obj.setup(
                        P=P_osqp,
                        q=q_update,
                        A=A_update,
                        l=l_update,
                        u=u_update,
                        verbose=False,
                        )

        osqp_result = osqp_obj.solve()

        # Extract the status string
        osqp_status_string = osqp_result.info.status

        # Display the status if it is anything other than success
        if (osqp_status_string != "solved"):
            raise ValueError("OSQP did not solve the problem, returned status = " + osqp_status_string)

        x_optimal = osqp_result.x

        s_opt, a_opt = rti.extract_values(x_optimal)


        trajectory["time"].append(t)
        trajectory["action"].append(a_opt[0])
        trajectory["state"].append(s_obs)
        trajectory["obs"].append(obs)
        trajectory["inf"].append(inf)

        a0 = a_opt[0].reshape(2,)

        positions.append(s_obs[0,0])
        deviations.append(s_obs[1,0])
        heading_angles.append(s_obs[2,0])
        velocities.append(s_obs[3,0] * 3.6)
        actions.append(a0)

        ground_truth_px.append(inf["ground_truth_px"][0])
        ground_truth_py.append(inf["ground_truth_py"][0])

        obs, _, terminated, truncated, inf = my_env.step(a0)

        t+= sampling_time
    
    print("Simulation finished!")
    plt.plot(ground_truth_px, ground_truth_py)

    plt.show()


if __name__ == "__main__":
    main()