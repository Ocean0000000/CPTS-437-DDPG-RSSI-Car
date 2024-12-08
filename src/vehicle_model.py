"""
a simple 2D rigid kinematics car model with euler stepping
"""
from config import *
import numpy as np

class Vehicle:
    """
    a simple 2D rigid body vehicle
    """

    def __init__(self, r: float, P: list[float], F: list[float], theta: float, throttle: float, steer: float, mix_method: str = "cosine",
                 throttle_multiplier: float = 1, friction_coefficient: float = 0.9) -> None:
        """
        initialize the vehicle

        :param r:                    width of vehicle
        :param P:                    vehicle center position vector
        :param theta:                the rotation of the vehicle
        :param throttle:             throttle setting of the car (clipped to [-1, 1])
        :param steer:                steer setting of the car (clipped to [-1, 1])
        :param mix_method:           method with which to mix a_l, a_r based on steer and throttle ('cosine' or 'linear')
        :param throttle_multiplier:  the throttle setting will be multiplied by this value to get the net a_l, a_r values
        :param friction_coefficient: for physics
        """

        if mix_method == "cosine":
            self.mix = self.cosine_mix
        elif mix_method == "linear":
            self.mix = self.linear_mix
        else:
            raise ValueError("unknown mix method")

        self.r = r
        self.P = np.array(P)
        self.theta = theta

        self.steer = steer
        self.throttle = throttle
        self.throttle_multiplier = throttle_multiplier

        self.friction_coefficient = friction_coefficient

        self.v_l = 0
        self.v_r = 0

        self.a_l, self.a_r = self.mix()

        self.i = np.array([np.cos(self.theta), np.sin(self.theta)]) # local x unit vector
        self.j = np.array([-1*np.sin(self.theta), np.cos(self.theta)])  # local y unit vector

        self.v_p_local = 0
        self.V_p = self.v_p_local * self.i + self.v_r * self.i  # velocity vector

        self._update_Ps()
        self.F = np.array(F)

    def linear_mix(self) -> tuple:
        """
        creates a linear mix based on throttle and steer
        """

        self.throttle = np.clip(self.throttle, a_min=-1, a_max=1)
        self.steer = np.clip(self.steer, a_min=-1, a_max=1)
            
        a_l = 1/2 * (1-self.steer) * self.throttle * self.throttle_multiplier - (abs(self.v_l**2) * self.friction_coefficient * np.sign(self.v_l))
        a_r = 1/2 * (1+self.steer) * self.throttle * self.throttle_multiplier - (abs(self.v_r**2) * self.friction_coefficient * np.sign(self.v_r))

        return a_l, a_r

    def cosine_mix(self) -> tuple:
        """
        creates a cosine mix based on throttle and steer
        """
        self.throttle = np.clip(self.throttle, a_min=-1, a_max=1)
        self.steer = np.clip(self.steer, a_min=-1, a_max=1)
        
        a_l = 1/2 * (np.cos( (2*np.pi*(self.steer+1)) / 4 ) + 1) * self.throttle * self.throttle_multiplier - (abs(self.v_l**2) * self.friction_coefficient * np.sign(self.v_l))
        a_r = 1/2 * (np.cos( (2*np.pi*(self.steer-1)) / 4 ) + 1) * self.throttle * self.throttle_multiplier - (abs(self.v_r**2) * self.friction_coefficient * np.sign(self.v_r))

        return a_l, a_r

    def reorient(self, gamma) -> None:
        """
        to be used in collisions
        """

        elastic_theta = 2*gamma - self.theta
        self.theta = elastic_theta
        self.theta = self.theta % (2*np.pi) # <- clip theta

        self.i = np.array([np.cos(self.theta), np.sin(self.theta)]) # local x unit vector
        self.j = np.array([-1*np.sin(self.theta), np.cos(self.theta)])  # local y unit vector

        #self.v_p_local = self.v_p_local * 0.5
        self.V_p = self.v_p_local * self.i + self.v_r * self.i  # velocity vector

    def step(self, dt: float) -> None:
        """
        take a step using simple euler.
        note that all vectors are in global coordinates.

        :param dt: time step size
        """

        self.a_l, self.a_r = self.mix()

        self.v_r = 10 * self.a_r * dt + self.v_r
        self.v_l = 10 * self.a_l * dt + self.v_l

        self.omega = (self.v_l - self.v_r)/self.r

        self.v_p_local = self.omega*(self.r/2)                  # magnitude of local velocity vector (w.r.t v_r)
        self.V_p = self.v_p_local * self.i + self.v_r * self.i  # velocity vector

        self.P = self.V_p * dt + self.P
        self._update_Ps()

        self.theta = self.omega * dt + self.theta
        self.theta = self.theta % (2*np.pi) # <- clip theta

        self.i = np.array([np.cos(self.theta), np.sin(self.theta)]) # local x unit vector
        self.j = np.array([-1*np.sin(self.theta), np.cos(self.theta)])  # local y unit vector

        return self.P


    def get_lines(self) -> list[dict[str, list]]:
        """
        returns a list of lines which, when plotted, draws the car.
        each line is stored as a dict containing "x" and "y" iterables
        """

        lines = []

        # car
        lines.append({"x": [self.P[0], self.P_f[0]], "y": [self.P[1], self.P_f[1]]})
        lines.append({"x": [self.P[0], self.P_b[0]], "y": [self.P[1], self.P_b[1]]})
        lines.append({"x": [self.P[0], self.P_l[0]], "y": [self.P[1], self.P_l[1]]})
        lines.append({"x": [self.P[0], self.P_r[0]], "y": [self.P[1], self.P_r[1]]})

        # target
        lines.append({"x": [self.F[0], self.F[0]], "y": [self.F[1] - self.r/2, self.F[1] + self.r/2]})
        lines.append({"x": [self.F[0] - self.r/2, self.F[0] + self.r/2], "y": [self.F[1], self.F[1]]})

        return lines

    
    def _update_Ps(self) -> None:
        """
        Update the positions of the parts of the car
        """

        self.P_l = self.P + self.r/2 * self.j
        self.P_r = self.P - self.r/2 * self.j

        if proportional:
            self.P_f = self.P + self.r/2 * self.i
            self.P_b = self.P - self.r/2 * self.i
        else:
            self.P_f = self.P + self.r * self.i
            self.P_b = self.P - self.r * self.i






