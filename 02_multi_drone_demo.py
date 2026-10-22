"""
Step 2 -- Formation take-off, square-waypoint flight and landing for 4 drones.

Demonstrates the core multi-vehicle pattern used in every multi-UAV script:
  1. issue non-blocking commands for ALL drones (each call returns a Future),
  2. join() all Futures to wait for the slowest one.

NED coordinates: x = north, y = east, z = DOWN (so z = -5 means 5 m above ground).
"""
import time

import airsim

DRONES = ["Drone1", "Drone2", "Drone3", "Drone4"]
HOVER_Z = -5.0        # metres, negative = above ground (NED)
SPEED = 3.0           # m/s

# Square patrol waypoints (NED, relative to each drone's spawn point)
WAYPOINTS = [
    (8.0, 0.0, HOVER_Z),
    (8.0, 8.0, HOVER_Z),
    (0.0, 8.0, HOVER_Z),
    (0.0, 0.0, HOVER_Z),
]


def wait_all(futures):
    for f in futures:
        f.join()


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Vehicles:", client.listVehicles())

    # --- 1. take control of every drone -------------------------------
    for d in DRONES:
        client.enableApiControl(True, vehicle_name=d)
        client.armDisarm(True, vehicle_name=d)

    # --- 2. parallel take-off ------------------------------------------
    wait_all([client.takeoffAsync(5, vehicle_name=d) for d in DRONES])
    time.sleep(2)

    # --- 3. climb to cruise altitude ------------------------------------
    wait_all([client.moveToZAsync(HOVER_Z, SPEED, vehicle_name=d) for d in DRONES])

    # --- 4. patrol the square, one drone tracks min-distance meanwhile --
    for leg, (tx, ty, tz) in enumerate(WAYPOINTS, start=1):
        futures = [client.moveToPositionAsync(tx, ty, tz, SPEED, vehicle_name=d)
                   for d in DRONES]
        while any(not f.is_completed() for f in futures):
            report_min_separation(client)
            time.sleep(0.5)
        wait_all(futures)
        print(f"[leg {leg}] all drones reached waypoint ({tx:.0f}, {ty:.0f}, {tz:.0f})")

    # --- 5. land, disarm, release control --------------------------------
    wait_all([client.landAsync(vehicle_name=d) for d in DRONES])
    wait_all([client.hoverAsync(vehicle_name=d) for d in DRONES])
    for d in DRONES:
        client.armDisarm(False, vehicle_name=d)
        client.enableApiControl(False, vehicle_name=d)
    print("[OK] Formation demo finished cleanly.")


def report_min_separation(client):
    """Print the smallest pair-wise distance -- the metric thesis hypothesis H1 needs."""
    positions = {}
    for d in DRONES:
        p = client.getMultirotorState(vehicle_name=d).kinematics_estimated.position
        positions[d] = (p.x_val, p.y_val, p.z_val)

    best = None
    names = list(positions)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = positions[names[i]], positions[names[j]]
            dist = sum((a[k] - b[k]) ** 2 for k in range(3)) ** 0.5
            if best is None or dist < best[0]:
                best = (dist, names[i], names[j])
    print(f"     min separation = {best[0]:6.2f} m  ({best[1]} <-> {best[2]})")


if __name__ == "__main__":
    main()
