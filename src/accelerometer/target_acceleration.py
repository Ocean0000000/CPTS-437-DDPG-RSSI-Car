"""
This module provides functions to calculate distances and compute the next state of an object based on its current position, velocity, and acceleration data from sensors.
Functions:
    calculate_distance(x_current, y_current, x_target, y_target):
        Calculates the Euclidean distance between the current position and a target position.
    comp_next_state(x_current, y_current, v_x_current, v_y_current, dt):
        Computes the next state of an object by updating its position and velocity based on current acceleration data from sensors and detects obstacles in front of it.
        
Uses Ueler integration to calculate the next state of the object based on the current position, velocity, and acceleration data from sensors. It also detects obstacles in front of the object and stops its motion if an obstacle is detected.
"""



import math
from pi_handler import get_a_x, get_a_y, get_front_distance

def calculate_distance(x_current, y_current, x_target, y_target):
    return math.sqrt((x_target - x_current) ** 2 + (y_target - y_current) ** 2)




def comp_next_state(x_current, y_current, v_x_current, v_y_current, dt):
    
    # Get current acceleration from sensors
    a_x_current = get_a_x()
    a_y_current = get_a_y()

    # Update velocity
    v_x_new = v_x_current + a_x_current * dt
    v_y_new = v_y_current + a_y_current * dt

    # Update position
    x_new = x_current + v_x_current * dt
    y_new = y_current + v_y_current * dt

    # Detect obstacles
    front_distance = get_front_distance()
    obstacle_detected = front_distance < 10  # Obstacle within 10 cm


    return {"x": x_new,"y": y_new,"v_x": v_x_new,"v_y": v_y_new,"obstacle_detected": obstacle_detected}
