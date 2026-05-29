import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from monovideoodometery import MonoVideoOdometery
import os
import argparse
import csv

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Process paths for image and pose data.')
    parser.add_argument('--img_path', type=str, default='./images', help='Path to the image directory')
    parser.add_argument('--pose_path', type=str, default='./pose', help='Path to the pose file')
    parser.add_argument('--csv_path', type=str, default='vo_output.csv', help='Path to the csv output file')
    parser.add_argument('--fov_deg', type=float, default=75.0, help='SuperSplat field of view in degrees')
    args = parser.parse_args()

    # Warning message if the default paths are used
    if args.img_path == './images' or args.pose_path == './pose':
        print("Warning: Using default paths './images' and './pose'. Specify --img_path and --pose_path to use custom paths.")

    img_path = args.img_path
    pose_path = args.pose_path
    csv_path = args.csv_path

    first_image = cv.imread(os.path.join(img_path, "000000.png"), 0)
    image_height, image_width = first_image.shape

    # calculate focal length from SuperSplat FoV
    fov_rad = np.radians(args.fov_deg)
    focal = image_width / (2.0 * np.tan(fov_rad / 2.0))
    pp = (image_width / 2.0, image_height / 2.0)
    R_total = np.zeros((3, 3))
    t_total = np.empty(shape=(3, 1))

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(21, 21),
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    # Create some random colors
    color = np.random.randint(0, 255, (5000, 3))

    vo = MonoVideoOdometery(img_path, pose_path, focal, pp, lk_params)
    traj = np.zeros(shape=(600, 800, 3))
    visual_scale = 25.0

    flag = False
    csv_folder = os.path.dirname(csv_path)
    if csv_folder != "":
        os.makedirs(csv_folder, exist_ok=True)

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "x", "y", "z",
        "true_x", "true_y", "true_z",
        "mse_error", "tracked_features",
        "delta_x", "delta_y", "delta_z",
        "true_step", "estimated_step",
    ])

    previous_true_position = None
    previous_estimated_position = None

    while vo.hasNextFrame():
        frame = vo.current_frame
        cv.imshow('frame', frame)
        k = cv.waitKey(1)
        if k == 27:  # Escape key to stop
            break

        if k == 121:  # 'y' key to toggle flow lines
            flag = not flag
            toggle_out = lambda flag: "On" if flag else "Off"
            print("Flow lines turned ", toggle_out(flag))
            mask = np.zeros_like(vo.old_frame)
            mask = np.zeros_like(vo.current_frame)

        vo.process_frame()

        print(vo.get_mono_coordinates())

        mono_coord = vo.get_mono_coordinates()
        true_coord = vo.get_true_coordinates()
        mse_error = np.linalg.norm(mono_coord - true_coord)

        current_estimated_position = np.array([mono_coord[0], mono_coord[1], mono_coord[2]])
        current_true_position = np.array([true_coord[0], true_coord[1], true_coord[2]])

        # Fehler zwischen geschaetzter und echter Position berechnen
        delta_x = mono_coord[0] - true_coord[0]
        delta_y = mono_coord[1] - true_coord[1]
        delta_z = mono_coord[2] - true_coord[2]

        # Bewegung seit dem letzten Frame berechnen
        if previous_true_position is None:
            true_step = 0.0
            estimated_step = 0.0
        else:
            true_step = np.linalg.norm(current_true_position - previous_true_position)
            estimated_step = np.linalg.norm(current_estimated_position - previous_estimated_position)

        print("MSE Error: ", mse_error)
        print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in mono_coord]))
        print("true_x: {}, true_y: {}, true_z: {}".format(*[str(pt) for pt in true_coord]))
        csv_writer.writerow([
            mono_coord[0],
            mono_coord[1],
            mono_coord[2],
            true_coord[0],
            true_coord[1],
            true_coord[2],
            mse_error,
            vo.n_features,
            delta_x,
            delta_y,
            delta_z,
            true_step,
            estimated_step,
        ])

        # Aktuelle Positionen fuer den naechsten Frame speichern
        previous_true_position = current_true_position
        previous_estimated_position = current_estimated_position

        # scale only for drawing
        draw_x = int(round(mono_coord[0] * visual_scale))
        draw_z = int(round(mono_coord[2] * visual_scale))
        true_x = int(round(true_coord[0] * visual_scale))
        true_z = int(round(true_coord[2] * visual_scale))

        traj = cv.circle(traj, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4)
        traj = cv.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

        cv.putText(traj, 'Actual Position:', (140, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(traj, 'Red', (270, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv.putText(traj, 'Estimated Odometry Position:', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(traj, 'Green', (270, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv.imshow('trajectory', traj)

    cv.imwrite("./images/trajectory.png", traj)
    csv_file.close()

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
