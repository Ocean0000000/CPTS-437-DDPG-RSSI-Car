"""
This script simulates the motion of an object towards a target position and visualizes the position, velocity, 
distance to the target, and acceleration over time using matplotlib.
Functions:
    main(): The main function that initializes the starting conditions, runs the simulation loop, updates the 
            object's state, controls the motors, and updates the plots in real-time.
Modules:
    matplotlib.pyplot: Used for plotting graphs.
    target_acceleration: Contains functions to calculate distance and compute the next state of the object.
    motor_control: Contains functions to control the motors. Would have used the pi_handler module From Joel however did not have acccess at the time.
The script performs the following steps:
1. Initializes the starting position, velocity, and target position.
2. Sets up the real-time plotting environment.
3. Enters a loop where it:
    - Computes the next state of the object.
    - Updates the current state.
    - Calculates the distance to the target.
    - Appends the data to lists for plotting.
    - Updates the plots in real-time.
    - Checks if the object has reached the target position and breaks the loop if so.
4. Finalizes and displays the plot.
"""





import time
import matplotlib.pyplot as plt
from target_acceleration import calculate_distance, comp_next_state
from motor_control import control_motors

#CoPilot Helped me with the actual plotting of the data

def main():
    x_current, y_current = 0, 0  # Starting position
    v_x_current, v_y_current = 0, 0  # Starting velocity
    x_target, y_target = 10, 10  # Target position
    dt = 0.1  # Time step

   
    time_data = []
    x_data, y_data = [], []
    v_x_data, v_y_data = [], []
    distance_data = []

   
    start_time = time.time()

    
    plt.ion()
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].set_title("Position (X and Y)")
    ax[0, 1].set_title("Velocity (X and Y)")
    ax[1, 0].set_title("Distance to Target")
    ax[1, 1].set_title("Acceleration (From Sensors)")
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 1].set_xlabel("Time (s)")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 1].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Position (m)")
    ax[0, 1].set_ylabel("Velocity (m/s)")
    ax[1, 0].set_ylabel("Distance (m)")
    ax[1, 1].set_ylabel("Acceleration (m/s²)")

    while True:
        
        elapsed_time = time.time() - start_time

        
        next_state = comp_next_state(x_current, y_current, v_x_current, v_y_current, dt)

        # Update current state
        x_current, y_current = next_state["x"], next_state["y"]
        v_x_current, v_y_current = next_state["v_x"], next_state["v_y"]

        
        distance_to_target = calculate_distance(x_current, y_current, x_target, y_target)

        # Get accelerations (assumed in `compute_next_state`)
        a_x, a_y = next_state.get("a_x", 0), next_state.get("a_y", 0)

        
        time_data.append(elapsed_time)
        x_data.append(x_current)
        y_data.append(y_current)
        v_x_data.append(v_x_current)
        v_y_data.append(v_y_current)
        distance_data.append(distance_to_target)

        # Control motors (this can also stop the loop if an obstacle is detected)
        # motor_status = control_motors(next_state)
        # if next_state["obstacle_detected"]:
        #     print("Obstacle detected. Stopping motion.")
        #     break

        # Update plots - This is where CoPilot helped me
        ax[0, 0].plot(time_data, x_data, label="X Position", color="blue")
        ax[0, 0].plot(time_data, y_data, label="Y Position", color="green")
        ax[0, 1].plot(time_data, v_x_data, label="X Velocity", color="orange")
        ax[0, 1].plot(time_data, v_y_data, label="Y Velocity", color="red")
        ax[1, 0].plot(time_data, distance_data, label="Distance", color="purple")
        ax[1, 1].plot(time_data, [0] * len(time_data), label="X Acceleration", color="cyan")  # Placeholder for a_x
        ax[1, 1].plot(time_data, [0] * len(time_data), label="Y Acceleration", color="magenta")  # Placeholder for a_y

        #It also helped here
        for a in ax.flat:
            a.legend()
            a.relim()
            a.autoscale_view()

        plt.pause(dt)

        # Break the loop if close enough to the target
        if distance_to_target < 0.1:
            print("Target reached. Stopping motion.")
            break

    # Finalize plot
    plt.ioff()
    plt.show()
    
    

if __name__ == "__main__":
    main()
