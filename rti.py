import numpy as np
from scipy import sparse, linalg


bicycle_model_parameters = {
    "Lf": 0.55 * 2.875,
    "Lr": 0.45 * 2.875,
    "m": 2000.0,
    "Iz": (1.0 / 12.0) * 2000.0 * (4.692**2 + 1.850**2),
    "Cm": (1.0 / 100.0) * (1.0 * 400.0 * 9.0) / 0.2286,
    "Cd": 0.5 * 0.24 * 2.2204 * 1.202,
    "delta_offset": 0 * np.pi / 180,
    "delta_request_max": 45 * np.pi / 180,
    "Ddelta_lower_limit": -45 * np.pi / 180,
    "Ddelta_upper_limit": 45 * np.pi / 180,
    "v_transition_min": 3.0,
    "v_transition_max": 5.0,
    "body_len_f": (0.55 * 2.875) * 1.5,
    "body_len_r": (0.45 * 2.875) * 1.5,
    "body_width": 2.50,
}

# Uncomment the road configuration you want to use: Easy, Medium, or Hard.
# Easy
road_elements_list = [
    {"type": "straight", "length": 500.0},
    {"type": "curved", "curvature": 1/500.0, "angle_in_degrees": 90.0},
    {"type": "straight", "length": 500.0},
]


# # Medium
# road_elements_list = [
#     {"type":"straight", "length":200.0},
#     {"type":"curved", "curvature":-1/200.0, "angle_in_degrees":80.0},
#     {"type":"straight", "length":400.0},
#     {"type":"curved", "curvature":1/300.0, "angle_in_degrees":80.0},
#     {"type":"straight", "length":300.0},
#     {"type":"curved", "curvature":-1/800.0, "angle_in_degrees":15.0},
#     {"type":"straight", "length":200.0},
#     {"type":"curved", "curvature":1/200.0, "angle_in_degrees":105.0},
#     {"type":"straight", "length":600.0},
# ]

# # Hard
# road_elements_list = [
#     {"type": "straight", "length": 500.0},
#     {"type": "curved", "curvature": 1 / 200.0, "angle_in_degrees": 180.0},
#     {"type": "straight", "length": 100.0},
#     {"type": "curved", "curvature": -1 / 200.0, "angle_in_degrees": 90.0},
#     {"type": "curved", "curvature": -1 / 200.0, "angle_in_degrees": 60.0},
#     {"type": "straight", "length": 100.0},
#     {"type": "curved", "curvature": 1 / 200.0, "angle_in_degrees": 60.0},
#     {"type": "straight", "length": 200.0},
#     {"type": "curved", "curvature": -1 / 400.0, "angle_in_degrees": 75.0},
#     {"type": "straight", "length": 100.0},
#     {"type": "curved", "curvature": 1 / 200.0, "angle_in_degrees": 25.0},
#     {"type": "curved", "curvature": 1 / 400.0, "angle_in_degrees": 120.0},
#     {"type": "curved", "curvature": 1 / 400.0, "angle_in_degrees": 120.0},
#     {"type": "straight", "length": 500.0},
# ]


numerical_integration_parameters = {
    "method": "rk4",
    "Ts": 0.05,
    "num_steps_per_Ts": 1,
}

py_init_min = 0.0
py_init_max = 0.0

v_init_min_in_kmh = 80.0
v_init_max_in_kmh = 80.0

initial_state_bounds = {
    "px_init_min": 0.0,
    "px_init_max": 0.0,
    "py_init_min": py_init_min,
    "py_init_max": py_init_max,
    "theta_init_min": 0.0,
    "theta_init_max": 0.0,
    "vx_init_min": v_init_min_in_kmh * (1.0 / 3.6),
    "vx_init_max": v_init_max_in_kmh * (1.0 / 3.6),
    "vy_init_min": 0.0,
    "vy_init_max": 0.0,
    "omega_init_min": 0.0,
    "omega_init_max": 0.0,
    "delta_init_min": 0.0,
    "delta_init_max": 0.0,
}


termination_parameters = {
    "speed_lower_bound": 0.0,
    "speed_upper_bound": (200.0 / 3.6),
    "distance_to_closest_point_upper_bound": 20.0,
    "reward_speed_lower_bound": 0.0,
    "reward_speed_upper_bound": 0.0,
    "reward_distance_to_closest_point_upper_bound": 0.0,
}


observation_parameters = {
    "should_include_ground_truth_px": "info",
    "should_include_ground_truth_py": "info",
    "should_include_ground_truth_theta": "info",
    "should_include_ground_truth_vx": "info",
    "should_include_ground_truth_vy": "info",
    "should_include_ground_truth_omega": "info",
    "should_include_ground_truth_delta": "info",
    "should_include_road_progress_at_closest_point": "obs",
    "should_include_vx_sensor": "obs",
    "should_include_distance_to_closest_point": "obs",
    "should_include_heading_angle_relative_to_line": "obs",
    "should_include_heading_angular_rate_gyro": "info",
    "should_include_closest_point_coords_in_body_frame": "info",
    "should_include_look_ahead_line_coords_in_body_frame": "info",
    "should_include_road_curvature_at_closest_point": "info",
    "should_include_look_ahead_road_curvatures": "info",
    "scaling_for_ground_truth_px": 1.0,
    "scaling_for_ground_truth_py": 1.0,
    "scaling_for_ground_truth_theta": 1.0,
    "scaling_for_ground_truth_vx": 1.0,
    "scaling_for_ground_truth_vy": 1.0,
    "scaling_for_ground_truth_omega": 1.0,
    "scaling_for_ground_truth_delta": 1.0,
    "scaling_for_road_progress_at_closest_point": 1.0,
    "scaling_for_vx_sensor": 1.0,
    "scaling_for_distance_to_closest_point": 1.0,
    "scaling_for_heading_angle_relative_to_line": 1.0,
    "scaling_for_heading_angular_rate_gyro": 1.0,
    "scaling_for_closest_point_coords_in_body_frame": 1.0,
    "scaling_for_look_ahead_line_coords_in_body_frame": 1.0,
    "scaling_for_road_curvature_at_closest_point": 1.0,
    "scaling_for_look_ahead_road_curvatures": 1.0,
    "vx_sensor_bias": 0.0,
    "vx_sensor_stddev": 0.1,
    "distance_to_closest_point_bias": 0.0,
    "distance_to_closest_point_stddev": 0.01,
    "heading_angle_relative_to_line_bias": 0.0,
    "heading_angle_relative_to_line_stddev": 0.01,
    "heading_angular_rate_gyro_bias": 0.0,
    "heading_angular_rate_gyro_stddev": 0.01,
    "closest_point_coords_in_body_frame_bias": 0.0,
    "closest_point_coords_in_body_frame_stddev": 0.0,
    "look_ahead_line_coords_in_body_frame_bias": 0.0,
    "look_ahead_line_coords_in_body_frame_stddev": 0.0,
    "road_curvature_at_closest_point_bias": 0.0,
    "road_curvature_at_closest_point_stddev": 0.0,
    "look_ahead_road_curvatures_bias": 0.0,
    "look_ahead_road_curvatures_stddev": 0.0,
    "look_ahead_line_coords_in_body_frame_distance": 100.0,
    "look_ahead_line_coords_in_body_frame_num_points": 10,
}


class RTI():
    def __init__(self, the_env, Ts, horizon, Q_weight, R_weight, 
                 Q_weight_ref, R_weight_ref, steering_kp, steering_kd):
        """
        Initializes the RTI class with environment, time step, horizon, 
        weights for Q and R matrices, and steering control parameters.

        Args:
            the_env: The environment object.
            Ts: Time step for discretization.
            horizon: Prediction horizon for MPC.
            Q_weight: Weight matrix for state cost.
            R_weight: Weight matrix for control cost.
            Q_weight_ref: Reference weight matrix for state cost.
            R_weight_ref: Reference weight matrix for control cost.
            steering_kp: Proportional gain for steering control.
            steering_kd: Derivative gain for steering control.
        """
        self.env = the_env
        self.Ts = Ts
        self.N = horizon

        self.Q = Q_weight
        self.R = R_weight

        self.Q_ref = Q_weight_ref
        self.R_ref = R_weight_ref

        self.kp = steering_kp
        self.kd = steering_kd

        self.prev_d = 0
    

    def k(self, progress):
        """
        Inputs:
            progress  : progress along the road (m)

        Return:
            road curvature at progress p_bar (1/m)
        """
        return self.env.unwrapped.road.convert_progression_to_curvature([progress])[0]




    def A_c(self, p_bar, d_bar, mu_bar, v_bar, F_bar, delta_bar):
        """
        Constructs the continuous-time linearization matrix A_c.

        Inputs:
            p_bar      : progress along the road (m)
            d_bar      : orthogonal distance to road (m)
            mu_bar     : heading relative to the path (rad)
            v_bar      : velocity (m/s)
            F_bar      : drive command (%)
            delta_bar  : steering angle (rad)

        Return:
            Ac        : 4x4 numpy matrix
        """

        kappa = self.k(p_bar)               # road curvature at p_bar
        Cd = bicycle_model_parameters["Cd"] # drag coefficient
        m = bicycle_model_parameters["m"]   # mass of the vehicle (kg)
        lr = bicycle_model_parameters["Lr"] # distance from rear axle to center of gravity (m)
        
        # common terms
        denom = (1 + d_bar * kappa)
        cos_mu = np.cos(mu_bar)
        sin_mu = np.sin(mu_bar)
        tan_delta = np.tan(delta_bar)
        sign_v = np.sign(v_bar)

        # A_c entries
        r0_c1 = -kappa * v_bar * cos_mu / (denom**2)
        r0_c2 = -v_bar * sin_mu / denom
        r0_c3 = cos_mu / denom

        r1_c2 = v_bar * cos_mu
        r1_c3 = sin_mu

        r2_c1 = (kappa**2) * v_bar * cos_mu / (denom**2)
        r2_c2 = kappa * v_bar * sin_mu / (denom)
        r2_c3 = (tan_delta / lr) - (kappa * cos_mu / denom)

        r3_c3 = 2 * Cd * v_bar * sign_v / m
        
        
        Ac = np.array([
            [0, r0_c1, r0_c2, r0_c3],
            [0,     0, r1_c2, r1_c3],
            [0, r2_c1, r2_c2, r2_c3],
            [0,     0,     0, r3_c3]
        ])
        
        return Ac
    
    def B_c(self, v_bar, delta_bar):
        """
        Constructs the continuous-time linearization matrix B_c.

        Inputs:
            v_bar      : velocity (m/s)
            delta_bar  : steering angle (rad)

        Return:
            Bc        : 4x2 numpy matrix
        """

        Cm = bicycle_model_parameters["Cm"] # coefficient of something
        m = bicycle_model_parameters["m"]   # mass of the vehicle (kg)
        lr = bicycle_model_parameters["Lr"] # distance from rear axle to center of gravity (m)
        
        cos_delta = np.cos(delta_bar)

        if cos_delta == 0: # catch division by zero
            return None
        
        sec_delta = 1/cos_delta

        # B_c entries
        r2_c1 = v_bar * (sec_delta**2) / lr

        r3_c0 = Cm/m
        
        Bc = np.array([
            [0,         0],
            [0,         0],
            [0,     r2_c1],
            [r3_c0,     0]
        ])
        
        return Bc
    
    def g_c(self, p_bar, d_bar, mu_bar, v_bar, F_bar, delta_bar):
        """
        Constructs the continuous-time linearization matrix g_c.

        Inputs:
            p_bar      : progress along the road (m)
            d_bar      : orthogonal distance to road (m)
            mu_bar     : heading relative to the path (rad)
            v_bar      : velocity (m/s)
            F_bar      : drive command (%)
            delta_bar  : steering angle (rad)

        Return:
            gc        : 4x1 numpy matrix
        """

        kappa = self.k(p_bar)               # road curvature at p_bar
        Cd = bicycle_model_parameters["Cd"] # drag coefficient
        Cm = bicycle_model_parameters["Cm"] #  coefficient
        m = bicycle_model_parameters["m"]   # mass of the vehicle (kg)
        lr = bicycle_model_parameters["Lr"] # distance from rear axle to center of gravity (m)
        
        # common terms
        denom = (1 + d_bar * kappa)
        cos_mu = np.cos(mu_bar)
        sin_mu = np.sin(mu_bar)
        tan_delta = np.tan(delta_bar)
        sign_v = np.sign(v_bar)

        # g_c entries
        r0_c0 = v_bar * cos_mu / denom
        r0_c1 = v_bar * sin_mu
        r0_c2 = (v_bar * tan_delta / lr) - (kappa * v_bar * cos_mu / denom)
        r0_c3 = (F_bar * Cm / m) - (Cd * (v_bar**2) * sign_v / m)
        
        
        term_1 = np.array([
            [r0_c0],
            [r0_c1],
            [r0_c2],
            [r0_c3]
        ])

        A_cont = self.A_c(p_bar, d_bar, mu_bar, v_bar, F_bar, delta_bar)

        B_cont = self.B_c(v_bar, delta_bar)

        s = np.array([
            [p_bar],
            [d_bar],
            [mu_bar],
            [v_bar]
        ])

        a = np.array([
            [F_bar],
            [delta_bar]
        ])

        gc = term_1 - np.matmul(A_cont, s) - np.matmul(B_cont, a)
        
        return gc

    def linearize(self, s, a):
        """
        Linearization of dynamics in continuous time.

        Inputs:
            s      : states, numpy array (4,1)
            a      : actions, numpy array (2,1)

        Return:
            A_lin_cont    
            B_lin_cont      
            g_lin_cont      
        """

        p_bar, d_bar, mu_bar, v_bar = s.reshape((4,))
        F_bar, delta_bar = a.reshape((2,))

        A_lin_cont = self.A_c(p_bar, d_bar, mu_bar, v_bar, F_bar, delta_bar)

        B_lin_cont = self.B_c(v_bar, delta_bar)

        g_lin_cont = self.g_c(p_bar, d_bar, mu_bar, v_bar, F_bar, delta_bar)

        return A_lin_cont, B_lin_cont, g_lin_cont 
    

    def discretize(self, A_lin, B_lin, g_lin):
        """
        Discretization of linearized dynamics.

        Inputs:
            A_lin      : numpy array (4,4)
            B_lin      : numpy array (4,2)
            g_lin      : numpy array (4,1)

        Return:
            A_disc    
            B_disc      
            g_disc      
        """

        # Determine the number of states and actions
        n_s = A_lin.shape[1]
        n_a = B_lin.shape[1]
        n_g = g_lin.shape[1]

        # Compute the discrete time dynamics matrices
        # > Stack the matrices
        M_cont = np.vstack([ np.hstack([A_lin, B_lin, g_lin]) , np.zeros((3,n_s+n_a+n_g), dtype=np.float32) ])
        # > Compute the matrix exponential
        M_disc = linalg.expm( M_cont * self.Ts)
        # > Extract the matrices
        A_disc = M_disc[0:n_s,0:n_s]
        B_disc = M_disc[0:n_s,n_s:(n_s+n_a)]
        g_disc = M_disc[0:n_s,(n_s+n_a):]

        return A_disc, B_disc, g_disc
    
    def dynamics_setup(self, s_list, a_list):
        """
        Linearize dynamics and return discretized matrices for all timesteps in the horizon

        Inputs:
            s_list      : list of states, each state is a numpy array (4,1)
            a_list      : list of actions, each action is a numpy array (2,1)

        Return:
            A_N    
            B_N      
            g_N      
        """

        A_N = []
        B_N = []
        g_N = []

        for i in range(len(a_list)): # there are N actions, and N+1 states
            A_cont, B_cont, g_cont = self.linearize(s_list[i], a_list[i])
            A_disc, B_disc, g_disc = self.discretize(A_cont, B_cont, g_cont)
            A_N.append(A_disc)
            B_N.append(B_disc)
            g_N.append(g_disc)

        return A_N, B_N, g_N
    

    def dyn_init(self, s, a):
        """
        Used only once!

        Constructs A_d, B_d, g_d. Only used for initialization of dynamics

        Inputs:
            s      : states, numpy array (4,1)
            a      : actions, numpy array (2,1)

        Return:
            sdot        : This only needs to be returned for the init function
            A_disc    
            B_disc      
            g_disc      
        """

        p_bar, d_bar, mu_bar, v_bar = s.reshape((4,))
        F_bar, delta_bar = a.reshape((2,))


        A_cont = self.A_c(p_bar, d_bar, mu_bar, v_bar, F_bar, delta_bar)

        B_cont = self.B_c(v_bar, delta_bar)

        g_cont = self.g_c(p_bar, d_bar, mu_bar, v_bar, F_bar, delta_bar)

        # Determine the number of states and actions
        n_s = A_cont.shape[1]
        n_a = B_cont.shape[1]
        n_g = g_cont.shape[1]

        # Compute the discrete time dynamics matrices
        # > Stack the matrices
        M_cont = np.vstack([ np.hstack([A_cont, B_cont, g_cont]) , np.zeros((3,n_s+n_a+n_g), dtype=np.float32) ])
        # > Compute the matrix exponential
        M_disc = linalg.expm( M_cont * self.Ts )
        # > Extract the matrices
        A_disc = M_disc[0:n_s,0:n_s]
        B_disc = M_disc[0:n_s,n_s:(n_s+n_a)]
        g_disc = M_disc[0:n_s,(n_s+n_a):]

        sdot = np.matmul(A_disc, s) + np.matmul(B_disc, a) + g_disc
        
        return sdot, A_disc, B_disc, g_disc


    def init_construct_ABg(self, s, a):
        """
        Used only once!

        Linearize dynamics and return discretized matrices for all timesteps in the horizon

        Inputs:
            s      : states, numpy array (4,1)
            a      : actions, numpy array (2,1)

        Return:
            A_N    
            B_N      
            g_N        
        """

        A_N = []
        B_N = []
        g_N = []
        
        states = [s]
        for n in range(self.N):
            sdot, A_disc, B_disc, g_disc = self.dyn_init(s, a)
            s = sdot
            A_N.append(A_disc)
            B_N.append(B_disc)
            g_N.append(g_disc)
            states.append(s)

        return A_N, B_N, g_N


    def create_reference(self, current_time, des_speed_ms, init_speed, current_state, begin=False):
        """
        Creates reference states and actions used within MPC. 
        
        *** Future update is to include the curvature to reduce speed ***

        Inputs:
            current_time      : current time of simulation
            des_speed_ms      : desired speed in m/s (assume desired speed is between 70-80km/h)
            init_speed        : added to account for first mpc solve which uses different speed
            current_state     : (4,1)

        Return:
            s_ref      : list of numpy array (4,1)
            a_ref      : list of numpy array (2,1)
        """
        s_ref = []
        a_ref = []

        progg = current_time*des_speed_ms

        ######################## VELOCITY SET - BEGIN ###############################
        current_p = current_state[0,0]
        curvature_close = self.k(current_p + 15)
        curvature_far = self.k(current_p + 100)  # ALSO THINK ABOUT VERY END!!!!



        if max(abs(curvature_close), abs(curvature_far)) >= (1/250):
            des_speed_ms = 10.55
            if not begin:
                init_speed = 10.55
        elif max(abs(curvature_close), abs(curvature_far)) >= (1/150):
            des_speed_ms = 5.55
            if not begin:
                init_speed = 5.55

        ######################## VELOCITY SET - END ###############################
        
        
        ref = np.array([[progg],[0],[0],[init_speed]])
        s_ref = [ref]

        ######################## STEERING ANGLE - BEGIN ###############################

        deriv = (current_state[1,0] - self.prev_d)/self.Ts
        
        delta_ref = -self.kp* (current_state[1,0]) - self.kd*(deriv) 

        ######################## STEERING ANGLE - END #################################
        

        a_ref.append(np.array([[0],[delta_ref]]))
        
        for n in range(1, self.N):

            dist = self.Ts*des_speed_ms 
            progg += dist
            r = np.array([[progg],[0],[0],[des_speed_ms]])
        
            s_ref.append(r)
            a_ref.append(np.array([[0],[delta_ref]]))
            
        self.prev_d = current_state[1,0]

        return s_ref, a_ref
    
    def mpc_setup(self):
        """
        Construct mpc problem.

        Note: This assumes that Q, R are global variables (change to class instances later)

        Inputs:

        Return:
            P_osqp    
        """
        n_s = 4
        # Objective function matrix
        QR = sparse.block_diag([self.Q, self.R])
        P_osqp  = sparse.block_diag([ sparse.kron(sparse.eye(self.N,format="csc"),QR), sparse.csc_matrix((n_s,n_s)) ],format="csc")

        return P_osqp


    def mpc_update(self, A_N, B_N, g_N, s_ref, a_ref, s0):
        """
        Update q, A l, u for mpc problem.
        Note: This assumes that a_lower, a_upper, Q, R, Q_ref and R_ref are global variables (change to class instances later)

        Inputs:
            A_N      :
            B_N      :
            g_N      :
            s_ref    :
            a_ref    : 
            s0       : numpy array (4,1)

        Return:
            q_updated    
            A_updated      
            l_updated
            u_updated      
        """

        n_s = 4
        n_a = 2

        # Box constraints
        a_lower = np.array([[-100.0], [-45*(np.pi/180)]])
        a_upper = np.array([[ 100.0], [ 45*(np.pi/180)]])

        ######################## CREATION OF AB MATRIX #########################
        helper_I = np.eye(self.N)
        result = []
        for i, (A, B) in enumerate(zip(A_N, B_N)):
            AB = np.hstack([A,B])
            kron_product = np.kron(helper_I[i,:], AB)
            result.append(kron_product)

        Atilde = np.vstack(result)
        A_tilde = sparse.csc_matrix(Atilde, shape=(n_s*self.N,(n_s+n_a)*self.N), dtype=np.float32)

        ######################## CREATION OF AB MATRIX #########################

        # Matrix for all time steps of the dynamics
        negI_zero_block = sparse.hstack([ -sparse.eye(n_s,format="csc") , sparse.csc_matrix((n_s,n_a)) ])
        zero_negI_col = sparse.vstack([sparse.csc_matrix(((self.N-1)*n_s,n_s)),-sparse.eye(n_s,format="csc")])
        ABI_stacked = \
            sparse.hstack([ A_tilde , sparse.csc_matrix((self.N*n_s,n_s)) ]) \
            + \
            sparse.hstack([ sparse.kron(sparse.eye(self.N,k=1,format="csc"),negI_zero_block) , zero_negI_col ])

        # Matrix for selecting the first state (I_tilde)
        s0_selector = sparse.hstack([ sparse.eye(n_s,format="csc"),sparse.csc_matrix((n_s,self.N*(n_s+n_a))) ])

        # Stacking the above two together
        Aeq = sparse.vstack([ABI_stacked, s0_selector])

        # Vector for equality constraint

        beq = np.vstack([np.vstack(g_N), s0])
            
        # Lower and upper bounds of box constraints
        x_lower = np.kron(np.ones((self.N,1),dtype=np.float32),a_lower)
        x_upper = np.kron(np.ones((self.N,1),dtype=np.float32),a_upper)

        Ia_blocks = sparse.hstack([sparse.csc_matrix((n_a, n_s)), sparse.eye(n_a,format="csc")])
        Ia = sparse.hstack([sparse.kron(sparse.eye(self.N, format="csc"), Ia_blocks), sparse.csc_matrix((n_a*self.N, n_s))])

        # Build the constraint matrix for OSQP format
        A_updated = sparse.vstack([ Aeq , Ia ],format="csc")

        # Build the constraint vectors for OSQP format
        l_updated = np.vstack([ beq , x_lower ])
        u_updated = np.vstack([ beq , x_upper ])

        ##################### q update #################################
        
        mpc_ref = []

        for i, elem in enumerate(s_ref):
                mpc_ref.append(-np.matmul(self.Q_ref,elem))
                mpc_ref.append(-np.matmul(self.R_ref,a_ref[i]))

        mpc_ref.append(-np.matmul(self.Q_ref,np.array([[0],[0],[0],[0]])))

        q_updated = np.vstack(mpc_ref)

        ##################### q update #################################
        

        return q_updated, A_updated, l_updated, u_updated


    def extract_values(self, optimal_vals):
        """
        Extracts the optimal state and control values from optimization problem

        Inputs:
            optimal_vals    : numpy array (64,1)

        Return:
            out_x           : list of numpy arrays (4,1)
            out_u           : list of numpy arrays (2,1)
        """
        n_s = 4
        n_a = 2

        out_x = []
        out_u = []
        for i in range(self.N):
            new_s = optimal_vals[i*(n_s+n_a): (i* (n_s+n_a)) + n_s].reshape((4,1))
            new_a = optimal_vals[(i*(n_s+n_a)) + n_s: (i* (n_s+n_a)) + n_s + n_a].reshape((2,1))
            out_x.append(new_s)
            out_u.append(new_a)

        new_s = optimal_vals[self.N*(n_s+n_a): (self.N* (n_s+n_a)) + n_s].reshape((4,1))
        out_x.append(new_s)

        return out_x, out_u


    def get_sate_observation(self, o_dict, i_dict):
        """
        After taking a step in environment, return the state as a vector
        
        TODO: Change the velocity to the sensor velocity

        Inputs:
            obs     : obersvations
            inf     : info dictionary

        Return:
            state observation, numpy array (4,1) 
        """
        p = o_dict["road_progress_at_closest_point"][0]
        d = o_dict["distance_to_closest_point"][0]
        mu = o_dict["heading_angle_relative_to_line"][0]
        v = o_dict["vx_sensor"][0]

        return np.array([[p],[d],[mu],[v]])