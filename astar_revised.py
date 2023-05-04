import matplotlib.pyplot as plt
import heapq
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import ttk

def is_adjacent_to_obstacle(grid, x, y):
    # Define the relative coordinates of the neighboring cells
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)] 
    
    # Iterate through the list of neighbors
    for dx, dy in neighbors:
        # Calculate the coordinates of the neighboring cell
        nx, ny = x + dx, y + dy
        
        # Check if the neighbor's coordinates are within the grid boundaries
        # and if the grid value at that position is greater than 0.5 (indicating an obstacle)
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx][ny] > 0.5:
            # If an obstacle is found in a neighboring cell, return True
            return True
            
    # If no obstacles are found in any neighboring cells, return False
    return False

def create_custom_grid(size, obstacles):
    # Create an empty grid with the given size
    grid = np.zeros((size, size))
    # Create an empty border grid with the same size
    border_grid = np.zeros((size, size))

    # Iterate through the list of obstacles
    for obstacle in obstacles:
        # Unpack the obstacle's parameters: center coordinates (center_x, center_y) and axes lengths (a, b)
        center_x, center_y, a, b = obstacle

        # Iterate through all the cells in the grid
        for x in range(size):
            for y in range(size):
                # Check if the current cell is inside the ellipse defined by the obstacle
                if ((x - center_x) ** 2 / a ** 2) + ((y - center_y) ** 2 / b ** 2) <= 1:
                    # If the cell is inside the obstacle, set its value to 1
                    grid[x][y] = 1
                # Check if the current cell is inside the border of the ellipse defined by the obstacle
                elif ((x - center_x) ** 2 / (a + 1) ** 2) + ((y - center_y) ** 2 / (b + 1) ** 2) <= 1:
                    # If the cell is inside the border, set its value to 1
                    border_grid[x][y] = 1

    # Define the border of the grid by setting its value to 1
    for i in range(100):
        for k in range(100):
            if (i == 0 and k != 0) or (i != 0 and k == 0) or (i == 99 and k != 99) or (i != 99 and k == 99):
                border_grid[i][k] = 1
                
    # Return the created grid and border grid
    return grid, border_grid

# Define a custom color map for visualization
cmap = ListedColormap(['red', 'white', 'yellow', 'black', 'green'])


# Define a function to update the heuristic based on the given heuristic_name
def update_heuristic(heuristic_name):
    # Access the global variable 'heuristic'
    global heuristic
    
    # Update the heuristic based on the heuristic_name
    if heuristic_name == "Manhattan Distance":
        heuristic = manhattan_distance
    elif heuristic_name == "Euclidean Distance":
        heuristic = euclidean_distance
    elif heuristic_name == "Chebyshev Distance":
        heuristic = chebyshev_distance
    elif heuristic_name == "Canberra Distance":
        heuristic = canberra_distance
    elif heuristic_name == "Jaccard Distance":
        heuristic = jaccard_distance
    elif heuristic_name == "Proposed Distance":
        heuristic = completely_custom
    elif heuristic_name == "Manhattan+Euclidean":
        heuristic = combined_heuristic

# Define a function to update the selected map based on the given map_name
def update_map(map_name):
    # Access the global variable 'map_selected'
    global map_selected
    
    # Update the map_selected variable with the given map_name
    map_selected = map_name


# Define a function to calculate the Manhattan distance between two points a and b
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Define a function to calculate the Euclidean distance between two points a and b
def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Define a function to calculate the Chebyshev distance between two points a and b
def chebyshev_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# Define a function to calculate the Canberra distance between two points a and b
def canberra_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sum(np.abs(a - b) / (np.abs(a) + np.abs(b)))

# Define a function to calculate the Jaccard distance between two points a and b
def jaccard_distance(a, b):
    a = np.array(a, dtype=bool)
    b = np.array(b, dtype=bool)
    intersection = np.logical_and(a, b)
    union = np.logical_or(a, b)
    return 1 - np.sum(intersection) / np.sum(union)

# Define a completely custom heuristic function for calculating the distance between two points a and b
def completely_custom(a, b):

     #---- Cosine Similarity  (1- Cosine Distance)
    # cos_one = (a[0] * b[0]) + (a[1] * b[1])
    # cos_two = np.sqrt(a[0]**2 + a[1]**2)
    # cos_three = np.sqrt(b[0]**2 + b[1]**2)
    # if cos_two==0 or cos_three == 0 :
    #     cos = 0 
    # else:
    #     cos = cos_one/(cos_two*cos_three)


    # Calculate the normalized vectors U1 and U2 for the given points a and b
    uu1 = (np.sqrt(a[0]**2 + a[1]**2))
    uu2 = (np.sqrt(b[0]**2 + b[1]**2))
    if uu1 == 0 and uu2 == 0:
        U1 = [0, 0]
        U2 = [0, 0]
    elif uu1 == 0 and uu2 != 0:
        U1 = [0, 0]
        U2 = b / uu2
    elif uu1 != 0 and uu2 == 0:
        U1 = a / uu1
        U2 = [0, 0]
    elif uu1 != 0 and uu2 != 0:
        U1 = a / uu1
        U2 = b / uu2

    # Calculate the new UU vector based on the absolute product of U1 and U2 elements
    UU = [abs(U1[0] * U2[1]), abs(U2[0] * U1[0])]

    # Calculate the coordinates of the destination point based on the selected map
    if map_selected == 'Map 1':
        end_x = round(14 * x_scaling_factor)
        end_y = round(13 * y_scaling_factor)
    elif map_selected == 'Map 2':
        end_x = round(8.25 * x_scaling_factor)
        end_y = round(15 * y_scaling_factor)
    else:
        end_x = round(7.75 * x_scaling_factor)
        end_y = round(16 * y_scaling_factor)

    # Calculate the deltas between UU and the destination point, and the start and destination points
    dx1 = UU[0] - end_x
    dy1 = UU[1] - end_y
    start_x = round(0.25 * x_scaling_factor)
    start_y = round(0.25 * y_scaling_factor)
    dx2 = start_x - end_x
    dy2 = start_y - end_y
    
    # Calculate the cost based on the calculated deltas
    Cost = abs(dx1 * dy2 - dx2 * dy1)
    # Cost = abs(dx1*dy1) + abs(dx2*dy2)

    return Cost

# Define a function to calculate a combined heuristic based on Manhattan and Euclidean distances between two points a and b
def combined_heuristic(a, b):
    manhattan_distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
    euclidean_distance = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    
    # Apply a random factor between 0.9 and 1.1
    random_factor = np.random.uniform(0.9, 1.1)
    
    return random_factor * (manhattan_distance + euclidean_distance) / 2




def astar(grid, start, end):
    # Define the possible neighbor positions relative to the current cell
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0),(-1, -1), (-1, 1), (1, -1), (1, 1)] # including neighbors
    # neighbors = [(0, 1), (1, 0), (0, -1), (-1,0)] # excluding neighbors
    # Initialize the open set with the start node and its f_score
    open_set = []
    heapq.heappush(open_set, (0, start))

    # Initialize the came_from dictionary to store the optimal path
    came_from = dict()

    # Initialize the gscore and fscore dictionaries with infinite values
    gscore = {(x, y): float("inf")
              for x in range(grid.shape[0]) for y in range(grid.shape[1])}
    fscore = {(x, y): float("inf")
              for x in range(grid.shape[0]) for y in range(grid.shape[1])}

    # Set the gscore of the start node to 0 and calculate its fscore
    gscore[start] = 0
    fscore[start] = heuristic(start, end)

    # Loop until the open set is empty
    while open_set:
        # Get the node with the lowest f_score
        _, current = heapq.heappop(open_set)

        # If the current node is the end node, reconstruct the path
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, fscore[end]

        # Iterate through the neighbors of the current node
        for dx, dy in neighbors:
            neighbor = current[0] + dx, current[1] + dy

            # Check if the neighbor is within the grid boundaries
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    # If the neighbor is an obstacle or adjacent to an obstacle, skip it
                    if grid[neighbor[0]][neighbor[1]] > 0.5 or is_adjacent_to_obstacle(grid, neighbor[0], neighbor[1]) or (neighbor[0] == 0 and neighbor[1] != 0) or (neighbor[0] != 0 and neighbor[1] == 0)or (neighbor[0] == 99 and neighbor[1] != 99) or (neighbor[0] !=99 and neighbor[1] == 99):
                        continue
                else:
                    continue
            else:
                continue

            # Calculate the tentative g_score for the neighbor

            # Calculate the tentative g_score for the neighbor
            tentative_gscore = gscore[current] + heuristic(current, neighbor)

            # If the tentative g_score is lower than the current g_score of the neighbor, update its values
            if tentative_gscore < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, end)
                heapq.heappush(open_set, (fscore[neighbor], neighbor))

    # If no path is found, return None
    return None, None

# Initialize a new tkinter window
root = tk.Tk()

# Set the title of the tkinter window
root.title("A* Algorithm - Heuristic Selection")

# Create the main frame with padding and add it to the root window
mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Configure the main frame's column and row to expand when the window is resized
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# Create a StringVar to store the heuristic choice
heuristic_choice = tk.StringVar()

# Create a StringVar to store the heuristic choice
map_choice = tk.StringVar()

# Create a label for the Heuristic dropdown menu
heuristic_label = ttk.Label(
    mainframe, text="Select Heuristic:", font=("Arial", 12))
heuristic_label.grid(column=0, row=0, sticky=tk.W)

# Create the dropdown menu for heuristic selection
heuristic_menu = ttk.OptionMenu(mainframe, heuristic_choice, "Manhattan Distance", "Euclidean Distance", "Chebyshev Distance",
                                "Canberra Distance", "Jaccard Distance", "Proposed Distance", "Manhattan+Euclidean", command=update_heuristic)
heuristic_menu.grid(column=1, row=0, sticky=(tk.W, tk.E))

# Create a label for the Map dropdown menu
map_label = ttk.Label(
    mainframe, text="Select Map:", font=("Arial", 12))
map_label.grid(column=0, row=1, sticky=tk.W)

# Create the dropdown menu for heuristic selection
map_menu = ttk.OptionMenu(mainframe, map_choice, "Map 1", "Map 2", "Map 3", command=update_map)
map_menu.grid(column=1, row=1, sticky=(tk.W, tk.E))


# Set the default heuristic to Manhattan Distance
heuristic_choice.set("Manhattan Distance")
heuristic = manhattan_distance

# Set the default map to map 1
map_choice.set("Map 1")
map_selected = 'Map 1'

# Create a button to run the A* algorithm and display the results
run_button = ttk.Button(mainframe, text="Run A*", command=root.quit)
run_button.grid(column=2, row=2, padx=10)

# Add some padding around widgets
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

# Start the GUI main loop
root.mainloop()
root.destroy()

grid_size = 100

if(map_selected == 'Map 1'):
    #Scaling Factor For Map 1
    x_scaling_factor = 100 / 15
    y_scaling_factor = 100 / 15
    obstacles = [
        # Map 1 obstacles -  Proposal image
        (2.5 * y_scaling_factor, 2 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (5 * y_scaling_factor, 2 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (7 * y_scaling_factor, 2 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (7.5 * y_scaling_factor, 2.5 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (4.75 * y_scaling_factor, 4 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (2 * y_scaling_factor, 5 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (6 * y_scaling_factor, 6 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (9 * y_scaling_factor, 6 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (0 * y_scaling_factor, 8 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (4 * y_scaling_factor, 8 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (8 * y_scaling_factor, 8 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (10 * y_scaling_factor, 8 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (10 * y_scaling_factor, 10 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (7 * y_scaling_factor, 10 * x_scaling_factor, .5* y_scaling_factor, .5 * x_scaling_factor),
        (5 * y_scaling_factor, 11 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (9 * y_scaling_factor, 12 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (11 * y_scaling_factor, 12 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        (7 * y_scaling_factor, 13 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
    ]
else:
    #Scaling Factor For Map 2 and Map 3
    x_scaling_factor = 100 / 18
    y_scaling_factor = 100 / 18
    if(map_selected == 'Map 2'):
        obstacles = [
            # Map 2 obstables - Example_2_1.jpg
            (2.25 * y_scaling_factor, 1.75 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (5 * y_scaling_factor, 2 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (4.75 * y_scaling_factor, 4 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (2 * y_scaling_factor, 5 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (2 * y_scaling_factor, 5 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (7 * y_scaling_factor, 5 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (4 * y_scaling_factor, 6 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (6 * y_scaling_factor, 7 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (8.5 * y_scaling_factor, 7 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (1 * y_scaling_factor, 8 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (3 * y_scaling_factor, 8 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (4 * y_scaling_factor, 8 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (8 * y_scaling_factor, 9 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (4 * y_scaling_factor, 10 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (6 * y_scaling_factor, 10 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (5 * y_scaling_factor, 11 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (10 * y_scaling_factor, 11 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (9 * y_scaling_factor, 12 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (5.5 * y_scaling_factor, 13 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (8 * y_scaling_factor, 13 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
            (6 * y_scaling_factor, 14 * x_scaling_factor, .5 * y_scaling_factor, .5 * x_scaling_factor),
        ]
    else:
        obstacles = [
            #Map 3 obstables - Example_3.jpg
            (2.25 * y_scaling_factor, 1.75 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (5 * y_scaling_factor, 2 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (4.75 * y_scaling_factor, 4 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (2 * y_scaling_factor, 5 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (2 * y_scaling_factor, 5 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (7 * y_scaling_factor, 5 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (4 * y_scaling_factor, 6 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (6 * y_scaling_factor, 7 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (8.5 * y_scaling_factor, 7 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (1 * y_scaling_factor, 8 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (3 * y_scaling_factor, 8 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (4 * y_scaling_factor, 8 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (8 * y_scaling_factor, 9 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (4 * y_scaling_factor, 10 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (6 * y_scaling_factor, 10 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (5 * y_scaling_factor, 11 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (10 * y_scaling_factor, 11 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (9 * y_scaling_factor, 12 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (5.5 * y_scaling_factor, 13 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (8 * y_scaling_factor, 13 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
            (6 * y_scaling_factor, 14 * x_scaling_factor, .2 * y_scaling_factor, .6 * x_scaling_factor),
        ]

#Creates output grid
grid, border_grid = create_custom_grid(grid_size, obstacles)

#-- Start same for all maps...
start_x = round(0.25*x_scaling_factor)
start_y = round(0.25*y_scaling_factor)

if map_selected == 'Map 1':
    # Map 1 end_x and end_y -  Proposal image
    end_x = round(14*x_scaling_factor)
    end_y = round(13*y_scaling_factor)
elif map_selected == 'Map 2':
    # Map 2 Example_2_1.jpg 
    end_x = round(8.25*x_scaling_factor)
    end_y = round(15*y_scaling_factor)
else:
    # Map 3 Example_3.jpg
    end_x = round(7.75*x_scaling_factor)
    end_y = round(16*y_scaling_factor)

#Setting start and end nodes
start, end = (start_x, start_y), (end_x, end_y)

# Call the astar function here
path, _ = astar(grid, start, end)

# Get the currently selected heuristic from the heuristic_choice variable
chosen_heuristic = heuristic_choice.get()

# Initialize an array with the same shape as 'grid' and filled with zeros
obstacle_colors = np.zeros_like(grid)

# Set the elements in 'obstacle_colors' to 1 where the corresponding 'grid' element is greater than 0.5
obstacle_colors[grid > 0.5] = 1


# If a path exists, update the grid to visualize the path
if path:
    # Set the path cells to have a value of 2
    for node in path:
        grid[node[0]][node[1]] = 2

    # Set the start and end cells to have values of 3 and 4, respectively
    grid[start[0]][start[1]] = 3
    grid[end[0]][end[1]] = 4

# Define the colormap for the obstacles
cmap = ListedColormap(['white', 'red'])

# Define a colormap for the borders
border_cmap = ListedColormap(['none', 'blue'])
border_image = plt.imshow(border_grid, cmap=border_cmap,
                          alpha=0.5)  # Display the border grid


# Display the grid with the 'viridis' colormap and set the alpha value to 0.5 for transparency
grid_image = plt.imshow(grid, cmap='viridis', alpha=0.5)


# Overlay the obstacle_colors on top of the grid with the defined cmap and set the alpha value to 0.5 for transparency
obstacle_image = plt.imshow(obstacle_colors, cmap=cmap, alpha=0.5)

# If a path exists, plot the path on the grid
if path:
    # Extract the x and y coordinates of the path
    path_x, path_y = zip(*path)
    # Plot the path with a yellow color, line style '-', circular markers, and a marker size of 5
    plt.plot(path_y, path_x, linewidth=2, color='yellow', linestyle='-',
             marker='o', markersize=5, markerfacecolor='yellow')
    
    
    # Calculate the total distance of the path
    total_distance = 0
    for i in range(len(path)-1):
        total_distance += np.sqrt((path[i][0]-path[i+1][0])
                                  ** 2 + (path[i][1]-path[i+1][1])**2)
        # total_distance += 1
        # total_distance = abs(path[i][0]-path[i+1][0]) + abs (path[i][1]-path[i+1][1])
        
    # Add a text box displaying the total distance at the top left corner of the graph
    plt.text(0, 97, f"Total Distance: {total_distance:.2f}",
             fontsize=16, bbox=dict(facecolor='white',alpha=0.5, pad=5) )


# Plot the start and end nodes separately
plt.plot(start_x,start_y, marker='o', markersize=10, color='black', label='Start')
plt.plot(end_y, end_x, marker='o', markersize=10, color='green', label='End')


# Create custom legend elements
legend_elements = [Line2D([0], [0], color='yellow', lw=2, label='Path'),
                   Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='black', markersize=8, label='Start'),
                   Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='g', markersize=8, label='End'),
                   Line2D([0], [0], marker='s', color='w',
                          markerfacecolor='red', markersize=8, label='Obstacle'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='#b09ed4', markersize=8, label='Buffer Zone')]  # Add buffer zone legend element

if map_selected == 'Map 1':
    """
    Ticks for Proposal Image
    """
    # Set the ticks for the grid
    x_ticks = np.linspace(0, 100, 16)
    y_ticks = np.linspace(0, 100, 16)

    x_tick_labels = list(range(0, 16))
    y_tick_labels = list(range(0, 16))

    plt.xticks(x_ticks, x_tick_labels)
    plt.yticks(y_ticks, y_tick_labels)

    ax = plt.gca()
    # Set the x and y limits for the plot
    ax.set_xlim(-2, 100)
    ax.set_ylim(-2, 100)
else:
    # Set the ticks for the grid
    x_ticks = np.linspace(0, 100, 19)
    y_ticks = np.linspace(0, 100, 19)

    x_tick_labels = list(range(0, 19))
    y_tick_labels = list(range(0, 19))

    plt.xticks(x_ticks, x_tick_labels)
    plt.yticks(y_ticks, y_tick_labels)

    ax = plt.gca()
    # Set the x and y limits for the plot, adjusting for the grid sizes and ratios
    ax.set_xlim(-2, 100)
    ax.set_ylim(-2, 88.56)

# Draw grid lines
plt.grid(which='both', color='k', linestyle='-', linewidth=1, alpha=0.1)

plt.title(
    f"A* Algorithm - Pathfinding with {chosen_heuristic} Heuristic",
    color='white', fontsize=18, pad=20)

# Adjust the layout of the plot
plt.tight_layout()

# Set the plot style to dark background
plt.style.use('dark_background')

# Add a legend to the plot
plt.legend(handles=legend_elements, bbox_to_anchor=
           (1, 1), loc='upper left', borderaxespad=0.0, fontsize=14)

# Show the plot
plt.show()



