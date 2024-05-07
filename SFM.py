import os
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objs as go
import gtsam
import tqdm

# Class to fetch and load images from a specified directory
class ImageFetcher:
    def __init__(self, path):
        self.images = []
        self.load_images(path)

    def load_images(self, path):
        # Sort files numerically and load each image, converting to RGB
        file_list = sorted(os.listdir(path), key=lambda x: int(os.path.splitext(x)[0]) if x.isdigit() else x)
        for filename in file_list:
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img_rgb)

# Class to handle feature detection and matching using SIFT
class FeatureMatcher:
    def __init__(self):
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)

    def detect_and_compute(self, images):
        # Detect and compute keypoints and descriptors for a list of images
        keypoints, descriptors = [], []
        for image in images:
            kp, des = self.sift.detectAndCompute(image, None)
            keypoints.append(kp)
            descriptors.append(des)
        return keypoints, descriptors

# Class to estimate poses using feature matches
class PoseEstimator:
    def __init__(self, images):
        # Initialize camera matrix based on the first image dimensions
        self.K = np.matrix([[1690, 0, images[0].shape[1] / 2],
                            [0, 1690, images[0].shape[0] / 2],
                            [0, 0, 1]])
        self.projections = [self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))]  
        self.points3d = []
        self.poses=[np.eye(4)]
        self.correspondences = {}

    def find_essential_matrix(self, des1, des2, kp1, kp2):
        # Match descriptors and compute the essential matrix
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(np.array(des1, dtype=np.float32), np.array(des2, dtype=np.float32), k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.0)
        pts1=pts1[mask.ravel()==1]
        pts2=pts2[mask.ravel()==1]
        U,S,Vt=np.linalg.svd(E)
        Ess=U@np.diag([1,1,0])@Vt
        return Ess, pts1, pts2
    
    def FetchPointsFromDescriptors(self,des1,des2,kp1,kp2):
        bf=cv2.BFMatcher(cv2.NORM_L2)
        descriptor_1 = np.array(des1, dtype=np.float32)
        descriptor_2 = np.array(des2, dtype=np.float32)
        matches=bf.knnMatch(descriptor_1,descriptor_2,k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7* n.distance:
                good_matches.append(m)
        pts1 =np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 =np.float32([kp2[m.trainIdx].pt for m in good_matches])


        return pts1,pts2

    def recover_pose_and_triangulate(self, E, src, dst):
        U, _, Vt = np.linalg.svd(E)
        W = np.matrix('0 -1 0; 1 0 0; 0 0 1')

        # Define possible rotations and translations
        rotations = [U @ W @ Vt, U @ W @ Vt, U @ W.T @ Vt, U @ W.T @ Vt]
        translations = [U[:, 2], -U[:, 2], U[:, 2], -U[:, 2]]

        modified_rotations = []
        modified_translations = []

        # Adjust rotations and translations based on their determinant
        for R, C in zip(rotations, translations):
            C_reshaped = C.reshape(-1, 1)
            if np.linalg.det(R) < 0:
                R = -1 * R
                C_reshaped = -1 * C_reshaped
            modified_rotations.append(R)
            modified_translations.append(C_reshaped)

        best_index =[]
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Initial projection matrix

        # Evaluate all rotation and translation pairs
        for R, C in zip(modified_rotations, modified_translations):
            P2 = self.K @ np.hstack((R, C))
            pts4d = cv2.triangulatePoints(P1, P2, src.T, dst.T)
            pts4d /= pts4d[3, :]  # Normalize the homogeneous coordinates
            points3d=pts4d[:3]
            valid_idx = (pts4d[2, :] > 0) & \
                    (pts4d[0, :] > -160) & (pts4d[0, :] < 160) & \
                    (pts4d[1, :] > -200) & (pts4d[1, :] < 200) & \
                    (pts4d[2, :] > -20) & (pts4d[2, :] < 500)
            pts3d = points3d[:, valid_idx]
            # Compute the Z-values
            Z_values = R[2] @ (pts3d - C.reshape(-1,1))
            positive_depth = np.sum(Z_values > 0)
            best_index.append(positive_depth)
            
            # Select the best projection based on the most positive Z-values
        finalindex=np.argmax(best_index)
        # Get the best rotation and translation
        best_R = modified_rotations[finalindex]
        best_T = modified_translations[finalindex].flatten()

        best_P2 = self.K @ np.hstack((best_R, best_T.reshape(3,1)))

        pose=np.eye(4)
        pose[:3,:3]=best_R
        pose[:3,3]=best_T
        self.poses.append(pose)
        # Store the best solution
        self.projections.append(best_P2)
        
        # Triangulate points using the best projection matrix
        best_pts4d = cv2.triangulatePoints(P1, best_P2, src.T, dst.T)
        

        best_pts4d /= best_pts4d[3, :]  # Normalize the points
        # valid_index=(best_pts4d[2, :] > 0) & \
                    # (best_pts4d[0, :] > -160) & (best_pts4d[0, :] < 160) & \
                    # (best_pts4d[1, :] > -200) & (best_pts4d[1, :] < 200) & \
                    # (best_pts4d[2, :] > -20) & (best_pts4d[2, :] < 500)
        
        # best_pts4d=best_pts4d[:,valid_index]
        self.points3d.append(best_pts4d)
        # dst=dst[valid_index]
        self.correspondences[1] = {tuple(dst[j]): tuple(best_pts4d[:, j][:3]) for j in range(len(dst))}

        return best_P2, best_pts4d
    

    def incremental_pose_estimation(self, descriptors, keypoints):
        for i in tqdm.tqdm(range(1, len(descriptors)-1), desc="Processing frames"):
            kp1, kp2 = self.FetchPointsFromDescriptors(descriptors[i], descriptors[i + 1], keypoints[i], keypoints[i + 1])
            new3dpts,new2dpts=self.find_correspondences(kp1,kp2,self.correspondences[i])


            _, rvec, tvec, inliers = cv2.solvePnPRansac(new3dpts, new2dpts, self.K, None)


            R, _ = cv2.Rodrigues(rvec)
            T = tvec
            P = self.K @ np.hstack((R, T))


            pts3d = cv2.triangulatePoints(self.projections[-1], P, kp1.T, kp2.T)
            pts3d /= pts3d[3, :]
            # valid_index = pts3d[2, :] >= 0
            # valid_pts3d = pts3d[:, valid_index]
            # valid_kp2 = kp2[valid_index]
            pose=np.eye(4)
            pose[:3,:3]=R
            pose[:3,3]=T.reshape(3)
            self.poses.append(pose)

            self.points3d.append(pts3d)
            self.projections.append(P)
            self.correspondences[i + 1] = {tuple(kp2[j]): tuple(pts3d[:,j][:3]) for j in range(len(kp2))}

    def find_correspondences(self, kp1, kp2, previous_correspondences):
        # Convert the keys of previous_correspondences into tuples for comparison
        previous_keys_tuples = [tuple(key) for key in previous_correspondences.keys()]

        # Lists to hold the found corresponding points and keypoints
        actual3d_pts = []
        next2d_pts = []

        # Iterate over the keypoints in the current frame
        for kp in range(len(kp1)):
            # Convert the keypoint position to a tuple for comparison
            kp1_tuple = tuple(kp1[kp])

            # Check if the tuple exists in the list of previous keys tuples
            if kp1_tuple in previous_keys_tuples:
                # Append the corresponding 3D point and the current keypoint to their respective lists
                actual3d_pts.append(previous_correspondences[kp1_tuple])
                next2d_pts.append(kp2[kp])
               
        return np.array(actual3d_pts), np.array(next2d_pts)
    
class StructureFromMotion:
    def __init__(self, image_folder, use_bundle_adjustment=False):
        # Initialize with the path to images and whether to use bundle adjustment
        self.image_folder = image_folder
        self.use_bundle_adjustment = use_bundle_adjustment
        self.fetcher = ImageFetcher(image_folder)
        self.matcher = FeatureMatcher()
        self.estimator = PoseEstimator(self.fetcher.images)

    def run(self):
    #     # Main execution flow of the Structure from Motion process
        keypoints, descriptors = self.matcher.detect_and_compute(self.fetcher.images)
        E, pts1, pts2 = self.estimator.find_essential_matrix(descriptors[0], descriptors[1], keypoints[0], keypoints[1])
        _, _ = self.estimator.recover_pose_and_triangulate(E, pts1, pts2)
        self.estimator.incremental_pose_estimation(descriptors, keypoints)
        # if self.use_bundle_adjustment:
            # self.perform_bundle_adjustment()
        self.visualize_results()

    def perform_bundle_adjustment(self):
        # Detailed implementation of the bundle adjustment process using GTSAM
        K_values = self.estimator.K
        Kmat = gtsam.Cal3_S2(K_values[0, 0], K_values[1, 1], 0, K_values[0, 2], K_values[1, 2])
        graph = gtsam.NonlinearFactorGraph()
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
        initial_pose = gtsam.Pose3(gtsam.Rot3(self.estimator.projections[0][:3, :3]), self.estimator.projections[0][:3, 3])
        graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, pose_noise))
        countL = 0
        for i, projection in enumerate(self.estimator.projections):
            for pt2d, pt3d in self.estimator.correspondences[i+1].items():
                point3d = gtsam.Point3(pt3d)
                measurement = gtsam.Point2(pt2d[0], pt2d[1])
                graph.add(gtsam.GenericProjectionFactorCal3_S2(measurement, measurement_noise, X(i), L(countL), Kmat))
                countL += 1
        initial_estimate = gtsam.Values()
        for i, projection in enumerate(self.estimator.projections):
            pose = gtsam.Pose3(gtsam.Rot3(projection[:3, :3]), projection[:3, 3])
            initial_estimate.insert(X(i), pose)
            for pt2d, pt3d in self.estimator.correspondences[i+1].items():
                initial_estimate.insert(L(countL), gtsam.Point3(pt3d))
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        result = optimizer.optimize()
        optimized_poses = []
        optimized_landmarks = []
        for i in range(len(self.estimator.projections)):
            optimized_pose = result.atPose3(X(i))
            optimized_poses.append(optimized_pose)
        for i in range(countL):
            optimized_landmark = result.atPoint3(L(i))
            optimized_landmarks.append(optimized_landmark)
        print('Initial error =', graph.error(initial_estimate))
        print('Final error =', graph.error(result))
        print('Optimized poses:', len(optimized_poses))
        print('Optimized landmarks:', len(optimized_landmarks))


    def visualize_results(self):
        # Collect all point cloud traces in a single plot
        def plot_point_clouds(points_list, title):
            traces = []
            for i, points in enumerate(points_list):
                if points.size == 0:
                    continue
                trace = go.Scatter3d(
                    x=points[0, :],  # X coordinates of points
                    y=points[1, :],  # Y coordinates of points
                    z=points[2, :],  # Z coordinates of points
                    mode='markers',
                    marker=dict(size=0.75, opacity=0.5, color='blue'),
                    name=f'Point Cloud {i + 1}'
                )
                traces.append(trace)
            layout = go.Layout(
                title=title,
                scene=dict(
        xaxis=dict(title='X Axis', range=[-160, 160]),  # Setting x-axis limits
        yaxis=dict(title='Y Axis', range=[-200, 200]),  # Setting y-axis limits
        zaxis=dict(title='Z Axis', range=[-20, 500])   # Setting z-axis limits
                )
            )
            fig = go.Figure(data=traces, layout=layout)
            fig.show()

        # Visualize camera poses
        def plot_camera_poses(poses, title):
            traces = []
            if len(poses) == 0:
                print("No camera poses to display.")
                return
            for i, pose in enumerate(poses):
                position = pose[:3, 3]
                x_axis = pose[:3, 0] * 5 + position
                y_axis = pose[:3, 1] * 5 + position
                z_axis = pose[:3, 2] * 5 + position
                traces.extend([
                    go.Scatter3d(x=[position[0], x_axis[0]], y=[position[1], x_axis[1]], z=[position[2], x_axis[2]],
                                mode='lines', line=dict(color='red', width=2), name=f'X Axis {i + 1}'),
                    go.Scatter3d(x=[position[0], y_axis[0]], y=[position[1], y_axis[1]], z=[position[2], y_axis[2]],
                                mode='lines', line=dict(color='green', width=2), name=f'Y Axis {i + 1}'),
                    go.Scatter3d(x=[position[0], z_axis[0]], y=[position[1], z_axis[1]], z=[position[2], z_axis[2]],
                                mode='lines', line=dict(color='blue', width=2), name=f'Z Axis {i + 1}')
                ])
            layout = go.Layout(title=title, scene=dict(
                xaxis=dict(title='X Axis'), 
                yaxis=dict(title='Y Axis'), 
                zaxis=dict(title='Z Axis')))
            fig = go.Figure(data=traces, layout=layout)
            fig.show()

        # Execute visualization functions
        if self.estimator.points3d:
            plot_point_clouds(self.estimator.points3d, '3D Reconstruction')
        if self.estimator.poses:
            plot_camera_poses(self.estimator.poses, 'Camera Poses')
        print(f"{len(self.estimator.poses)}")


        # if self.use_bundle_adjustment:
        #     if hasattr(self, 'optimized_poses') and hasattr(self, 'optimized_landmarks'):
        #         plot_point_cloud(np.hstack(self.optimized_landmarks), 'Optimized 3D Reconstruction', color='red')
        #         plot_camera_poses(self.optimized_poses, 'Optimized Camera Trajectories')

def main():
    # Set the path to the directory containing your images
    image_directory = '/home/nitin/Deep-Learning-based-Feature-Matching/sfm_image'
    
    # Initialize the StructureFromMotion system
    sfm_system = StructureFromMotion(image_directory, use_bundle_adjustment=False)
    
    # Run the Structure from Motion process
    sfm_system.run()

if __name__ == "__main__":
    main()
