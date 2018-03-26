1. Quick start

The quickest method to run the motion-planner is to start the simulator first, then in a terminal type:
$ python motion_planning.py

The planner will:
 (1) read lat0, lon0 from colliders into floating point values
 (2) set home position to (lon0, lat0, 0)
 (3) retrieve current global position
 (4) convert to current local position
 (5) convert start position to current position rather than map center
 (6) randomly choose a free location from the grid, set it as the goal location
 (7) adapt to set goal as latitude / longitude position
 (8) perform A* algorithm to search a path
 (9) prune the number of turning points of the path

If you want to see the path in the grid map found by the A* algorithm, type
$ python motion_planning.py --debug=True

We provide three levels of pruning degree. To see different pruning performance, type
$ python motion_planning.py --prune_level=0|1|2

prunde_level=0 does not performe any pruning method.
prunde_level=1 performes a simple pruning method: if the same action (turning UP, DOWN, LEFT, RIGHT) is
        consequtively executed, their will be merged into one action.
prunde_level=2 performes full pruning，using the following method: Given a path with turning points (p1, p2, ..., pn),
        (1) p_1 is a turning point
        (2) if p_i is a turning point, the next turning point p_{i+k} is the maximum k such that the line segment from
            p_i to p_{i+j} does not intersect any obstacle. To reduce the complexity, we draw the convex hull of the
            path, and create polygons from grid. The search space of obstacle is all those polygons which intersect with
            the convex hull.
            prunde_level=2 is the default value used by the motion planner.





