  mport cv2
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# Parameters for feature detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
def track_features(prev_frame, curr_frame, prev_points):
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, None, **lk_params)
    good_points = curr_points[status == 1]
    return good_points
def compute_depth(prev_points, curr_points, focal_length):
    depth = np.linalg.norm(prev_points - curr_points, axis=1)
    depth = (focal_length * 0.1) / depth  # Adjusting the scale factor (0.1) as per requirements
    return depth
def reconstruct_3d(points, depth, camera_matrix):
    # Convert 2D points to homogeneous coordinates
    points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    # Invert camera matrix to obtain the transformation from homogeneous 2D to 3D
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    # Transpose the points_homogeneous array
    points_homogeneous_transposed = points_homogeneous.T
    # Reshape the depth array to have shape (N, 1) where N is the number of points
    depth_reshaped = depth.reshape(-1, 1)
    # Transform 2D points to 3D using depth and camera matrix
    points_3d_homogeneous = np.matmul(camera_matrix_inv, points_homogeneous_transposed)
    points_3d_homogeneous = points_3d_homogeneous.T
    # Scale the 3D points using depth
    points_3d = depth_reshaped * points_3d_homogeneous[:, :3]
    return points_3d
def visualize_3d(points_3d):   fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], cmap='hsv')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open the camera.")
        return
    _, frame = cap.read()
    # Get image dimensions
    if frame is None:
        print("Failed to capture a frame.")
        cap.release()
        cv2.destroyAllWindows()
        return
    height, width, _ = frame.shape
    # Initialize variables
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    focal_length = width  # Assuming focal length = width of the image
    # Initialize camera matrix
    camera_matrix = np.array([[focal_length, 0, width / 2],
                              [0, focal_length, height / 2],
                              [0, 0, 1]])
    points_3d = np.empty((0, 3))  # Initialize an empty array to store 3D points
    while True:
        # Capture frame...
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture a frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Track features and compute depth...
        curr_points = track_features(prev_gray, gray, prev_points)
        depth = compute_depth(prev_points, curr_points, focal_length)
        # Reconstruct 3D points...
        curr_points = curr_points.reshape(-1, 1, 2)  # Reshape to (N, 1, 2) for better compatibility
        points_3d_curr = reconstruct_3d(curr_points, depth, camera_matrix)
        # Concatenate the current 3D points with the previous ones
        points_3d = np.concatenate((points_3d, points_3d_curr), axis=0)
        # Visualize 3D points
        visualize_3d(points_3d)
        # Update previous frame and points
        prev_gray = gray.copy()
        prev_points = curr_points
        cv2.imshow("Video", prev_gray)                                           # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
# Run the main function
if __name__ == "__main__":
    main()
