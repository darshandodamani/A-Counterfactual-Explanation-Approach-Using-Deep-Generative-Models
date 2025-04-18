# location: carla_dataset_collection/collect_images.py
import carla
import random
import time
import os
import numpy as np
import cv2
import csv
import logging
import argparse

logging.basicConfig(level=logging.INFO)

# Utility Functions
def connect_to_carla():
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        return client
    except Exception as e:
        logging.error(f"Failed to connect to CARLA: {e}")
        raise

def setup_world(client, town_name):
    logging.info(f"Loading {town_name}...")
    world = client.load_world(town_name)

    # Synchronous mode setup
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    logging.info("World loaded successfully.")

    # Return world and traffic manager
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    return world, tm

def spawn_npc_vehicles(world, tm, num_vehicles=30):
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()

    vehicles_list = []
    random.shuffle(spawn_points)

    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_blueprints)
        transform = spawn_points[i]
        npc = world.try_spawn_actor(bp, transform)
        if npc:
            npc.set_autopilot(True, tm.get_port())
            vehicles_list.append(npc)
    
    logging.info(f"Spawned {len(vehicles_list)} NPC vehicles.")
    return vehicles_list


def setup_vehicle(world, vehicle_bp_name="vehicle.audi.a2"):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(vehicle_bp_name)
    spawn_points = world.get_map().get_spawn_points()
    start_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, start_point)
    vehicle.set_autopilot(True)
    logging.info("Vehicle spawned and autopilot enabled.")
    return vehicle

def setup_camera(world, vehicle, image_size_x=160, image_size_y=80, fov=125):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(image_size_x))
    camera_bp.set_attribute("image_size_y", str(image_size_y))
    camera_bp.set_attribute("fov", str(fov))
    camera_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    logging.info("Camera sensor attached to the vehicle.")
    return camera

# Dataset Collection Function
def collect_images(output_dir, town_name, image_size):
    client = connect_to_carla()
    world, tm = setup_world(client, town_name)  # Updated to return tm
    npc_vehicles = spawn_npc_vehicles(world, tm, num_vehicles=30)  # Pass tm to spawn NPCs
    vehicle = setup_vehicle(world)
    camera = setup_camera(world, vehicle, image_size[0], image_size[1])
    
    image_array = None
    total_frames_collected = 0
    total_frames_saved = 0
    total_frames_skipped = 0

    def process_image(image):
        nonlocal image_array
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        image_array = array[:, :, :3]

    camera.listen(process_image)
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', town_name + '_dataset')
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"{town_name}_data_log.csv")

    try:
        frame = 0
        with open(csv_filename, "w", newline="") as csvfile:
            fieldnames = ["image_filename", "steering", "throttle", "brake"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while frame < 15000:
                world.tick()
                time.sleep(0.2)
                total_frames_collected += 1

                if image_array is not None:
                    image_name = f"{town_name}_{frame:06d}.png"
                    image_path = os.path.join(output_dir, image_name)
                    if cv2.imwrite(image_path, image_array):
                        logging.info(f"Saved {image_name}")
                        total_frames_saved += 1
                    else:
                        logging.error(f"Failed to save image {image_name}")
                        raise RuntimeError(f"Failed to save image {image_name}")

                    control = vehicle.get_control()
                    writer.writerow({
                        "image_filename": image_name,
                        "steering": control.steer,
                        "throttle": control.throttle,
                        "brake": control.brake,
                    })
                    logging.info(f"Saved control data for {image_name}")
                    
                    # Verify CSV update
                    csvfile.flush()
                    os.fsync(csvfile.fileno())
                    logging.info(f"CSV file updated successfully for {image_name}")
                    frame += 1
                else:
                    logging.warning("Image not available yet, skipping frame.")
                    total_frames_skipped += 1

    finally:
        if camera:
            camera.stop()
            camera.destroy()
        if vehicle:
            vehicle.destroy()
        for npc in npc_vehicles:  # Destroy all NPC vehicles
            npc.destroy()
        logging.info("All actors destroyed.")

        # Log summary for transparency
        logging.info(f"Total frames collected: {total_frames_collected}")
        logging.info(f"Total frames saved: {total_frames_saved}")
        logging.info(f"Total frames skipped: {total_frames_skipped}")
        logging.info(f"Total labeled frames: {frame}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA Dataset Collection Script")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save collected dataset.")
    parser.add_argument("--town_name", type=str, default="Town03", help="CARLA town name to load.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[160, 80], help="Image size (width height).")
    args = parser.parse_args()

    collect_images(args.output_dir, args.town_name, args.image_size)
