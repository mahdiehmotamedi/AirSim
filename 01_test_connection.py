"""
Step 1 -- Verify that the simulator is running and the Python API is reachable.

Run order: start the simulator (BlocksV2 / CityEnviron / your UE project) first,
then:  python 01_test_connection.py
"""
import sys

try:
    import airsim
except ImportError:
    sys.exit("airsim package not found. Run:  pip install airsim msgpack-rpc-python")


def main():
    # Default host is 127.0.0.1:41451. For a remote machine use:
    #   airsim.MultirotorClient(ip="192.168.1.50", port=41451)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OK] Connected to simulator.")

    vehicles = client.listVehicles()
    print("[OK] Vehicles in scene:", vehicles)
    if not vehicles:
        print("[WARN] No vehicles found. Check the 'Vehicles' section of settings.json")

    for name in vehicles:
        state = client.getMultirotorState(vehicle_name=name)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        print(
            f"     {name:10s}  pos=(x={pos.x_val:7.2f}, y={pos.y_val:7.2f}, z={pos.z_val:7.2f})"
            f"  vel=(vx={vel.x_val:5.2f}, vy={vel.y_val:5.2f}, vz={vel.z_val:5.2f})"
        )

    print("\n[OK] API version check passed. You are ready to run 02_multi_drone_demo.py")


if __name__ == "__main__":
    main()
