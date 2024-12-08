"""
Overview:
Involves controlling a robotic system to navigate towards a target position using accelerometer data. 
The main script initializes the current state of the robot, calculates the distance to the target, and iteratively 
updates the robot's position and velocity until it reaches the target. The script is also to include obstacle detection from Joels code
Modules:
- target_acceleration: Contains functions to calculate distance and compute the next state of the robot.
- pi_handler: Contains functions to handle motor control and other Raspberry Pi related operations.
Functions:
- main(): The main function that initializes the robot's state, calculates the path to the target, and controls the 
    robot's movement until the target is reached or an obstacle is detected.
"""




from target_acceleration import calculate_distance, comp_next_state
from pi_handler import *




def main():
    x_current, y_current = 0, 0 
    v_x_current, v_y_current = 0, 0  
    x_target, y_target = 10, 10  
    dt = 0.1  # Time step


    
    # insert the model here
    # Initialize the current state
    # Initialize the project
    

    
    
    #start 
    distance_to_target = calculate_distance(x_current, y_current, x_target, y_target)
    # Initialize array to store the path
    path = [(x_current, y_current)]
    
    

    while distance_to_target > 0.1:  # Loop until the target is reached or very close
        
        next_state = comp_next_state(x_current, y_current, v_x_current, v_y_current, dt)

        
        #this is where the model would go
        # Control motors based on the next state
        #motor_status = control_motors(next_state)

        # Print current state for debugging
        # print(f"Position: ({next_state['x']:.2f}, {next_state['y']:.2f}) | "
        #       f"Velocity: ({next_state['v_x']:.2f}, {next_state['v_y']:.2f}) | "
        #       f"Motor Status: {motor_status}")

        # Check for obstacles
        if next_state["obstacle_detected"]:
            #this is where model decides
            break

        # Update current state for the next iteration
        x_current, y_current = next_state["x"], next_state["y"]
        v_x_current, v_y_current = next_state["v_x"], next_state["v_y"]

        # Update the distance to the target
        distance_to_target = calculate_distance(x_current, y_current, x_target, y_target)
        path.append((x_current, y_current))
        





    # Final cleanup
    print("targeting complete")
    control_motors({"v_x": 0, "v_y": 0, "obstacle_detected": True})

if __name__ == "__main__":
    main()
