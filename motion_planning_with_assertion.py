import argparse
import time
import msgpack
import random
import copy
from enum import Enum, auto
from shapely.geometry import LineString, Polygon
import numpy as np

from planning_utils import a_star, heuristic, create_grid_polygons, prune_path, grid_to_local, local_to_grid
import waypoint_data
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local, local_to_global


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()
    WAYUPDATE = auto()


class MotionPlanning(Drone):
    def __init__(self, connection, mp_method, pruneLevel, debug, nodeId):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.last_targets = []
        self.num_of_steps = 0
        self.total_steps = 15
        self.TARGET_ALTITUDE = 1
        self.SAFETY_DISTANCE = 5
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}
        self._mp_method = mp_method
        self._prune_level = pruneLevel
        self._debug = debug
        self._nodeId = nodeId

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        print("local position: ", self.local_position[0], self.local_position[1], 'state:', self.flight_state)
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            self.num_of_steps += 1
            print("num of steps: ", self.num_of_steps)
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()
            elif self.num_of_steps > 3:
                print(self.last_targets[:10])
                print(self.waypoints[:10])
                self.flight_state = States.WAYUPDATE
                print('cleanup waypoints in simulator   ')
                index, length = 0, len(self.last_targets)
                if index < length:
                    point = self.last_targets[index]
                    while np.linalg.norm(point[0:2] - self.local_position[0:2]) >= 1.0:
                        if index == length:
                            break
                        else:
                            index += 1
                            point = self.last_targets[index]
                    while np.linalg.norm(point[0:2] - self.local_position[0:2]) < 1.0:
                        if index == length:
                            break
                        else:
                            index += 1
                            point = self.last_targets[index]
                self.clear_waypoints()
                raise_num = 1
                lst = self.last_targets[:index]
                lst.reverse()
                print(lst)
                print(self.last_targets[:10])
                print(self.waypoints[:10])

                self.waypoints = lst + [self.target_position] +self.waypoints
                print(self.waypoints[:10])
                self.waypoints = [[ele[0], ele[1], ele[2] + raise_num, ele[3]] for ele in self.waypoints]
                self.raise_transition(raise_num)
                self.waypoint_transition()

        elif self.flight_state == States.WAYUPDATE:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
            index, length  = 0, len(self.last_targets)
            if index < length:
                point = self.last_targets[index]
                while np.linalg.norm(point[0:2] - self.local_position[0:2]) >= 1.0:
                    if index == length:
                        break
                    else:
                        index += 1
                        point = self.last_targets[index]
                while np.linalg.norm(point[0:2] - self.local_position[0:2]) < 1.0:
                    if index == length:
                        break
                    else:
                        index += 1
                        point = self.last_targets[index]
            self.clear_waypoints()
            raise_num = 1
            lst = self.last_targets[:index]
            lst.reverse()
            self.waypoints = lst+ [self.target_position] + self.waypoints
            print(self.waypoints[:10])
            self.waypoints = [[ele[0], ele[1], ele[2] + raise_num, ele[3]] for ele in self.waypoints]
            self.raise_transition(raise_num)
            self.waypoint_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    print("armed...., planning path....")
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition", self.target_position[2])
        self.takeoff(self.target_position[2])

    def raise_transition(self, delta):
        self.flight_state = States.WAYUPDATE
        print("raising transition", self.target_position[2] + delta)
        self.disarming_transition()
        time.sleep(1)
        self.arming_transition()
        self.send_waypoints()
        self.takeoff(self.target_position[2] + delta)

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.last_targets.insert(0, self.target_position)
        # print('last targets:', self.last_targets)
        self.target_position = self.waypoints.pop(0)
        self.num_of_steps = 0
        if len(self.target_position) == 3:
            print(self.target_position, self.waypoints)

        print('target position', self.target_position)
        self.send_waypoints()
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2],
                          self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        # data = msgpack.dumps(self.waypoints)
        # print(self.waypoints[:2], type(self.waypoints[:2]))
        data = msgpack.packb(self.waypoints[:2], strict_types=True)
        # data = msgpack.packb([self.target_position], strict_types=True)
        # print("_master?", type(self.connection._master), dir(self.connection._master))
        # print(self.connection._master.waypoint_request_list_send())
        self.connection._master.write(data)

    def clear_waypoints(self):
        print("clear those waypoints which have been sent to the simulator ...")
        self.connection._master.waypoint_clear_all_send()

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")

        self.target_position[2] = self.TARGET_ALTITUDE

        # Set self.waypoints
        if debug:
            self.waypoints = waypoint_data.waypoints
            print("*waypoints*", waypoint_data.waypoints)
            # TODO: send waypoints to sim
            self.send_waypoints()
            return

        # TODO: read lat0, lon0 from colliders into floating point values
        with open('colliders.csv') as ifh:
            for ln in ifh:
                if ln.startswith('lat0'):
                    lst = ln.split(' ')
                    lat0, lon0 = [float(ele) for ele in [lst[1].strip(','), lst[3]]]
                    break

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        # TODO: retrieve current global position
        global_position = np.array([self._longitude, self._latitude, self._altitude])

        # TODO: convert to current local position using global_to_local()
        self._north, self._east, self._down = global_to_local(global_position, self.global_home)
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset, polygons = create_grid_polygons(data,
                                                                         self.TARGET_ALTITUDE,
                                                                         self.SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # show_grid(grid)
        # Define starting point on the grid (this is just grid center)
        # grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center
        north_start, east_start, down_start = self.local_position
        # grid_start = (int(north_start - north_offset), int(east_start - east_offset))
        grid_start = local_to_grid(self.local_position, northOffset=north_offset, eastOffset=east_offset)
        print('start grid', grid_start)
        # grid_start = (315, 445)
        # Set goal as some arbitrary position on the grid
        # grid_goal = (-north_offset + 10, -east_offset + 30)

        # TODO: adapt to set goal as latitude / longitude position and convert
        # randomly choose a free cell in the grid as the target
        # transform it into latitude/longitude position, and print out
        freeCells = [list(cell) for cell in np.argwhere(grid == 0)]
        random.shuffle(freeCells)
        grid_goal = tuple(freeCells[0])
        # grid_goal = (750, 284)
        print("*grid_goal*", grid_goal)
        # grid_goal = (7-north_offset, 789-east_offset)
        global_position = local_to_global((grid_goal[0] + north_offset, grid_goal[1] + east_offset, 0),
                                          self.global_home)
        print("goal positon:", global_position)

        if self._mp_method == "a_star":
            # Run A* to find a path from start to goal
            # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
            # or move to a different search space such as a graph (not done here)
            print('Local Start and Goal: ', grid_start, grid_goal)
            print('performing a-star search....')
            path, cost, prunedPath = a_star(grid, heuristic, grid_start, grid_goal, polygons,
                                            debug=self._debug)
            if cost == -1:
                return
            # TODO: prune path to minimize number of waypoints
            # TODO (if you're feeling ambitious): Try a different approach altogether!

            if self._prune_level == 2:
                path = prune_path(prunedPath, polygons, debug=self._debug, nodeId=self._nodeId)
                waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), self.TARGET_ALTITUDE, 0] for p in path]
            elif self._prune_level == 1:
                print("level 1 pruning is performed.")
                print("Turning points are reduced from {} to {}".format(len(path), len(prunedPath)))
                waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), self.TARGET_ALTITUDE, 0]
                             for p in prunedPath]
            elif self._prune_level == 0:
                waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), self.TARGET_ALTITUDE, 0] for p in path]

        elif self._mp_method == "simplest":
            pass
        elif self._mp_method == "medial_axis":
            pass
        elif self._mp_method == "pb_map":
            pass
        elif self._mp_method == "voronoi":
            pass

        # Set self.waypoints
        self.waypoints = waypoints
        print("*waypoints*", waypoints)
        # TODO: send waypoints to sim
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    """
    Usage:
    $ python motion_planning.py --prune_level 0|1|2
                                --debug True
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--method', type=str, default="a_star", help='planning methods')
    parser.add_argument('--prune_level', type=int, default=2,
                        help='level of prune for a_star result, 0: no prune, 1: partial prune, 2: full prune')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--nodeId', type=int, default=-1, help='the i-th node on the path')
    args = parser.parse_args()
    print(args)
    method, pruneLevel, debug, nodeId = args.method, args.prune_level, args.debug, args.nodeId
    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn, method, pruneLevel, debug, nodeId)
    time.sleep(1)

    drone.start()