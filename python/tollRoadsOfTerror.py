







import math
from collections import deque

def genGrid(filename):
    # Read the entire content of the file
    with open(filename, 'r') as file:
        # Read all data and replace commas with spaces
        data = file.read().replace(',', ' ')
    
    # Split the data into a list of integers
    elements = list(map(int, data.split()))
    
    # Calculate the dimension N of the square grid
    N = int(math.sqrt(len(elements)))
    
    if N * N != len(elements):
        raise ValueError("The number of elements in the file is not a perfect square, cannot form an NXN grid.")
    
    # Create the grid as an NXN list
    grid = [elements[i * N:(i + 1) * N] for i in range(N)]
    
    return grid




import heapq

def p1_min_cost_route(grid):
    # Define the directions for moving and their corresponding labels
    directions = [(0, 1, 'r'), (1, 0, 'd'), (0, -1, 'l'), (-1, 0, 'u')]
    
    # Initialize the starting point
    start = (0, 0)
    end = (len(grid) - 1, len(grid[0]) - 1)
    
    # Priority queue for Dijkstra's algorithm with (cost, row, col, path)
    priority_queue = [(grid[start[0]][start[1]], start[0], start[1], [])]
    
    # Dictionary to track the minimum cost to reach each cell
    min_cost = {start: grid[start[0]][start[1]]}
    
    # Store the paths to the end
    path_to_end = None

    # Dijkstra's loop to find the path from start to end
    while priority_queue:
        current_cost, x, y, path = heapq.heappop(priority_queue)
        
        # If we've reached the end point, save the path and cost
        if (x, y) == end:
            path_to_end = path
            break  # Stop after finding the shortest path to the end
        
        # Explore the four possible directions
        for dx, dy, direction in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within grid bounds
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                # Calculate the new cost
                new_cost = current_cost + grid[nx][ny]
                
                # Check if we found a cheaper way to reach (nx, ny)
                if (nx, ny) not in min_cost or new_cost < min_cost[(nx, ny)]:
                    min_cost[(nx, ny)] = new_cost
                    heapq.heappush(priority_queue, (new_cost, nx, ny, path + [directions]))
    
    if not path_to_end:
        return None, []  # If no path to end is found

    # Extract the cost to the end
    cost_to_end = min_cost[end]
    
    # Reset the priority queue for finding the path from end back to start
    priority_queue = [(grid[end[0]][end[1]], end[0], end[1], [])]
    min_cost = {end: grid[end[0]][end[1]]}

    path_back_to_start = None

    # Dijkstra's loop to find the path from end back to start
    while priority_queue:
        current_cost, x, y, path = heapq.heappop(priority_queue)
        
        # If we've reached the start point, save the path
        if (x, y) == start:
            path_back_to_start = path
            break  # Stop after finding the shortest path back to start
        
        # Explore the four possible directions
        for dx, dy, direction in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within grid bounds
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                # Calculate the new cost
                new_cost = current_cost + grid[nx][ny]
                
                # Check if we found a cheaper way to reach (nx, ny)
                if (nx, ny) not in min_cost or new_cost < min_cost[(nx, ny)]:
                    min_cost[(nx, ny)] = new_cost
                    heapq.heappush(priority_queue, (new_cost, nx, ny, path + [direction]))

    # Return the total cost and the complete path (start to end, then end back to start)
    if path_back_to_start:
        complete_path = path_to_end + list(reversed(path_back_to_start))
        total_cost = cost_to_end + min_cost[start]
        return total_cost, complete_path
    else:
        return None, []  # If no path back to start is found









import heapq
from collections import defaultdict

def find_min_cost_route(grid, traversed_cells, taxes, player_key):
    # Directions for moving and their corresponding labels
    directions = [(0, 1, 'r'), (1, 0, 'd'), (0, -1, 'l'), (-1, 0, 'u')]
    
    # Initialize the starting point
    start = (0, 0)
    end = (len(grid) - 1, len(grid[0]) - 1)
    
    # Priority queue for Dijkstra's algorithm with (cost, row, col, path)
    priority_queue = [(grid[start[0]][start[1]], start[0], start[1], [(0, 0)])]
    
    # Dictionary to track the minimum cost to reach each cell
    min_cost = {start: grid[start[0]][start[1]]}
    
    # Store the path to the end
    path_to_end = None

    # Dijkstra's loop to find the path from start to end
    while priority_queue:
        current_cost, x, y, path = heapq.heappop(priority_queue)
        
        # If we've reached the end point, save the path and cost
        if (x, y) == end:
            path_to_end = path
            break  # Stop after finding the shortest path to the end
        
        # Explore the four possible directions
        for dx, dy, direction in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within grid bounds
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                # Calculate the new cost, including additional route tax for traversed cells
                tax = traversed_cells[(nx, ny)]
                new_cost = current_cost + grid[nx][ny] + tax
                
                # Check if we found a cheaper way to reach (nx, ny)
                if (nx, ny) not in min_cost or new_cost < min_cost[(nx, ny)]:
                    min_cost[(nx, ny)] = new_cost
                    # Append the current move's indices to the path
                    heapq.heappush(priority_queue, (new_cost, nx, ny, path + [(nx, ny)]))
    
    if not path_to_end:
        return None, []  # If no path to end is found

    # Extract the cost to the end
    cost_to_end = min_cost[end]
    
    # Reset the priority queue for finding the path from end back to start
    priority_queue = [(grid[end[0]][end[1]], end[0], end[1], [end])]
    min_cost = {end: grid[end[0]][end[1]]}

    path_back_to_start = None

    # Dijkstra's loop to find the path from end back to start
    while priority_queue:
        current_cost, x, y, path = heapq.heappop(priority_queue)
        
        # If we've reached the start point, save the path
        if (x, y) == start:
            path_back_to_start = path
            break  # Stop after finding the shortest path back to start
        
        # Explore the four possible directions
        for dx, dy, direction in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within grid bounds
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                # Calculate the new cost, including additional route tax for traversed cells
                tax = traversed_cells[(nx, ny)]
                new_cost = current_cost + grid[nx][ny] + tax
                
                # Check if we found a cheaper way to reach (nx, ny)
                if (nx, ny) not in min_cost or new_cost < min_cost[(nx, ny)]:
                    min_cost[(nx, ny)] = new_cost
                    heapq.heappush(priority_queue, (new_cost, nx, ny, path + [(nx, ny)]))

    # Combine the paths from start to end and back to start
    if path_back_to_start:
        complete_path = path_to_end + list(reversed(path_back_to_start))
        # Store the path as indices for the player
        taxes[player_key] = complete_path
        total_cost = cost_to_end + min_cost[start]
        return total_cost, complete_path
    else:
        return None, []  # If no path back to start is found

def simulate_players(grid):
    # Initialize route tax tracker and player move tracker
    traversed_cells = defaultdict(int)
    taxes = defaultdict(list)  # Stores the cells each player has traversed
    player_results = []
    
    # Simulate each player
    for player_id in range(2, 6):
        player_key = f'p{player_id}'
        
        # Find the optimal route for the current player
        total_cost, path = find_min_cost_route(grid, traversed_cells, taxes, player_key)
        
        # Record the result for the current player
        player_results.append({'player': player_key, 'total_cost': total_cost, 'path': path})
        
        # Update the route tax tracker for the current player's path
        for (x, y) in path:
            traversed_cells[(x, y)] += 1
        
        # Cross-reference current player's path with previous players
        for (x, y) in path:
            for previous_player in taxes:
                if previous_player != player_key and (x, y) in taxes[previous_player]:
                    # Increment the cost of the previous player
                    for result in player_results:
                        if result['player'] == previous_player:
                            result['total_cost'] += 1
    
    return player_results









board = genGrid('./TXTs/Game_Seed_Qual_Round_22.txt')
# print(board)

cost, moves = p1_min_cost_route(board)
print("Total Cost:", cost)
print("Sequence of Moves:", moves)

results = simulate_players(board)
for result in results:
    print(f"Player {result['player']} Total Cost: {result['total_cost']}, Path: {result['path']}\n")
    
    



