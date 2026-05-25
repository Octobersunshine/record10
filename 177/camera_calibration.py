import cv2
import numpy as np
import os
import glob
from typing import Tuple, Optional
from config import Config


class StereoCalibration:
    def __init__(self, config: Config):
        self.config = config
        self.left_camera_matrix = None
        self.left_dist_coeffs = None
        self.right_camera_matrix = None
        self.right_dist_coeffs = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
        
    def calibrate_single_camera(self, 
                                image_pattern: str, 
                                chessboard_size: Tuple[int, int],
                                square_size: float) -> Tuple[np.ndarray, np.ndarray, float]:
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        obj_points = []
        img_points = []
        
        images = glob.glob(image_pattern)
        gray_shape = None
        
        for image_path in images:
            img = cv2.imread(image_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret:
                obj_points.append(objp)
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                img_points.append(corners_refined)
        
        if len(obj_points) < 10:
            raise ValueError(f"Not enough valid calibration images: {len(obj_points)}/10")
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray_shape, None, None
        )
        
        return camera_matrix, dist_coeffs, ret
    
    def calibrate_stereo(self, left_images_pattern: str, right_images_pattern: str):
        chessboard_size = self.config.CHESSBOARD_SIZE
        square_size = self.config.CHESSBOARD_SQUARE_SIZE
        
        print("Calibrating left camera...")
        self.left_camera_matrix, self.left_dist_coeffs, left_ret = \
            self.calibrate_single_camera(left_images_pattern, chessboard_size, square_size)
        print(f"Left camera calibration reprojection error: {left_ret:.4f}")
        
        print("Calibrating right camera...")
        self.right_camera_matrix, self.right_dist_coeffs, right_ret = \
            self.calibrate_single_camera(right_images_pattern, chessboard_size, square_size)
        print(f"Right camera calibration reprojection error: {right_ret:.4f}")
        
        print("Performing stereo calibration...")
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        obj_points = []
        img_points_left = []
        img_points_right = []
        
        left_images = sorted(glob.glob(left_images_pattern))
        right_images = sorted(glob.glob(right_images_pattern))
        
        for left_path, right_path in zip(left_images, right_images):
            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)
            
            if img_left is None or img_right is None:
                continue
                
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
            
            if ret_left and ret_right:
                obj_points.append(objp)
                
                corners_left_refined = cv2.cornerSubPix(
                    gray_left, corners_left, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                corners_right_refined = cv2.cornerSubPix(
                    gray_right, corners_right, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                img_points_left.append(corners_left_refined)
                img_points_right.append(corners_right_refined)
        
        img_size = gray_left.shape[::-1]
        
        ret, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            obj_points, img_points_left, img_points_right,
            self.left_camera_matrix, self.left_dist_coeffs,
            self.right_camera_matrix, self.right_dist_coeffs,
            img_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        print(f"Stereo calibration reprojection error: {ret:.4f}")
        
        self._compute_rectification_maps(img_size)
        
    def _compute_rectification_maps(self, img_size: Tuple[int, int]):
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.left_camera_matrix, self.left_dist_coeffs,
            self.right_camera_matrix, self.right_dist_coeffs,
            img_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_dist_coeffs,
            self.R1, self.P1, img_size, cv2.CV_32FC1
        )
        
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_dist_coeffs,
            self.R2, self.P2, img_size, cv2.CV_32FC1
        )
        
        self.config.Q = self.Q
        
    def rectify_images(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.left_map1 is None:
            self._initialize_from_config(left_img.shape[:2])
            
        left_rectified = cv2.remap(left_img, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def _initialize_from_config(self, img_shape: Tuple[int, int]):
        img_size = (img_shape[1], img_shape[0])
        
        self.left_camera_matrix = self.config.LEFT_CAMERA_MATRIX
        self.left_dist_coeffs = self.config.LEFT_DIST_COEFFS
        self.right_camera_matrix = self.config.RIGHT_CAMERA_MATRIX
        self.right_dist_coeffs = self.config.RIGHT_DIST_COEFFS
        self.R = self.config.R
        self.T = self.config.T
        
        self._compute_rectification_maps(img_size)
    
    def load_calibration(self, filepath: str):
        data = np.load(filepath)
        self.left_camera_matrix = data['left_camera_matrix']
        self.left_dist_coeffs = data['left_dist_coeffs']
        self.right_camera_matrix = data['right_camera_matrix']
        self.right_dist_coeffs = data['right_dist_coeffs']
        self.R = data['R']
        self.T = data['T']
        self.Q = data['Q']
        
        img_size = (640, 480)
        self._compute_rectification_maps(img_size)
        
        print(f"Calibration loaded from {filepath}")
    
    def save_calibration(self, filepath: str):
        np.savez(filepath,
                 left_camera_matrix=self.left_camera_matrix,
                 left_dist_coeffs=self.left_dist_coeffs,
                 right_camera_matrix=self.right_camera_matrix,
                 right_dist_coeffs=self.right_dist_coeffs,
                 R=self.R, T=self.T, Q=self.Q)
        print(f"Calibration saved to {filepath}")
