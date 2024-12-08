"""
This is a filler file, its used in place of Joels pi_handler module.

This module provides motor control functionality for an accelerometer-based system.
Functions:
    control_motors(motion_state): Controls the motors based on the provided motion state.
The `control_motors` function takes a dictionary `motion_state` as input, which contains information about the current motion state, including velocity and obstacle detection. If an obstacle is detected, it stops all motors. Otherwise, it calculates the throttle values for the motors based on the velocity and sets the motor throttle accordingly.
Dependencies:
    - pi_handler: A module that provides functions to set motor throttle and stop all motors.
"""



from pi_handler import set_throttle, stop_all_motors




def control_motors(motion_state):
    if motion_state["obstacle_detected"]:
        stop_all_motors()
        #need a change direction function ask ocean and diego what they want
        return "Motors stopped due to obstacle."

    # Calculate throttle from velocity (example logic, adjust as needed)
    throttle_x = min(max(motion_state["v_x"], -1), 1)  
    throttle_y = min(max(motion_state["v_y"], -1), 1) 

    # Set motor throttle
    set_throttle(1, throttle_x)  
    set_throttle(2, throttle_y) 

    return f"Motors running with throttle: X={throttle_x:.2f}, Y={throttle_y:.2f}"



