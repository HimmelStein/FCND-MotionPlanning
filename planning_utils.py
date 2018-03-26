from enum import Enum
from queue import PriorityQueue
from shapely.geometry import Polygon, LineString, Point
from sklearn.neighbors import KDTree
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


def find_waypoints_from_to(path, waypoint, to=()):
    idx0 = path.index(waypoint)
    idx1 = path.index(to)
    return path[idx0:idx1+1]


def next_waypoint_of(waypoint, path=[]):
    idx = path.index(waypoint)
    if idx < len(path)-2:
        return path[idx+1]
    else:
        return []


def before_waypoint_of(waypoint, path=[]):
    idx = path.index(waypoint)
    if idx >0:
        return path[idx-1]
    else:
        return []


def prune_path(path, polygons, debug=False):
    """
    :param path:
    :param polygons:
    :return:
    """
    startIdx = 0
    ptStart = path[startIdx]
    result = [ptStart]
    print('length of original path:', len(path))
    while True:
        idx = startIdx + 1
        nextIdx = startIdx + 1
        # while idx <= len(path) - 1:
        for i in range(len(path)-idx):
            if idx + i > len(path)-1:
                continue
            ptx = path[idx+i]
            ln = LineString([ptStart[::-1], ptx[::-1]])
            canConnect = True
            for polygon in polygons:
                if polygon.crosses(ln):
                    canConnect = False
                    if debug and startIdx ==5:
                        plot_collider(polygon, ln, polygons, path)
                        print(polygon.boundary.xy)
                        print(ln.xy)
                    break # for polygon

            if canConnect:
                nextIdx = idx + i

        startIdx = nextIdx
        ptStart = path[startIdx]
        result.append(ptStart)
        if startIdx == len(path) -1:
            break
    print('length of pruned path:', len(result))
    return result


def create_grid_polygons(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min + 1)))
    east_size = int(np.ceil((east_max - east_min + 1)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Initialize polygons
    polygons = []

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
            plg = Polygon([(obstacle[2]+1,obstacle[0]+1),
                           (obstacle[3],obstacle[0]+1),
                           (obstacle[3], obstacle[1]),
                           (obstacle[2]+1,obstacle[1])])
            polygons.append(plg)
    return grid, int(north_min), int(east_min), polygons


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions


def a_star(grid, h, start, goal, ploygons, debug=False):
    """
    Given a grid and heuristic function returns
    the lowest cost path from start to goal.
    """

    path, prunedPath = [], []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            # Get the new vertexes connected to the current vertex
            for a in valid_actions(grid, current_node):
                next_node = (current_node[0] + a.delta[0], current_node[1] + a.delta[1])
                new_cost = current_cost + a.cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))
                    branch[next_node] = (new_cost, current_node, a)

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        prunedPath.append(goal)
        lastAction = branch[n][2]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
            if branch[n][2] != lastAction:
                prunedPath.append(branch[n][1])
                lastAction = branch[n][2]
        path.append(branch[n][1])
        prunedPath.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    if debug:
        path0 = [ele[::-1] for ele in path]
        plot_path(path0, ploygons)
    return path[::-1], path_cost, prunedPath[::-1]


def heuristic(position, goal_position):
    return np.sqrt((position[0] - goal_position[0])**2 + (position[1]-goal_position[1])**2)


def show_grid(grid):
    plt.imshow(grid, cmap='Greys', origin='lower')
    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.show()


def plot_path(path, ploygons):
    for pg in ploygons:
        x0, y0 = pg.exterior.xy
        plt.plot(x0, y0, 'blue', alpha=0.5)
    x,y = [ele[0] for ele in path], [ele[1] for ele in path]
    plt.plot(x,y, 'red', alpha=0.5)
    plt.show()


def plot_collider(plg, line, ploygons, path):
    for pg in ploygons:
        x0, y0 = pg.exterior.xy
        plt.plot(x0, y0, 'blue', alpha=0.5)
    x,y = plg.exterior.xy
    x1,y1 = line.xy
    plt.plot(x,y, 'blue', alpha=0.5)
    plt.plot(y1,x1, 'red', alpha=0.5)
    path = [ele[::-1] for ele in path]
    x,y = [ele[0] for ele in path], [ele[1] for ele in path]
    plt.plot(x, y, 'black', alpha=0.5)
    plt.show()


