import numpy as np

class Config:
    VIDEO_LEFT_PATH = "videos/left_camera.mp4"
    VIDEO_RIGHT_PATH = "videos/right_camera.mp4"
    
    CALIBRATION_DIR = "calibration"
    CALIBRATION_FILE = "calibration/stereo_calibration.npz"
    CALIBRATION_IMAGE_PATTERN = "calibration_images/calib_{cam}_{idx}.jpg"
    CHESSBOARD_SIZE = (9, 6)
    CHESSBOARD_SQUARE_SIZE = 25.0
    
    INSECT_MIN_AREA = 10
    INSECT_MAX_AREA = 500
    MOTION_THRESHOLD = 25
    BACKGROUND_SUBTRACTOR_HISTORY = 500
    
    STEREO_MATCHER = "SGBM"
    STEREO_MIN_DISPARITY = 0
    STEREO_NUM_DISPARITIES = 128
    STEREO_BLOCK_SIZE = 15
    EPIPOLAR_THRESHOLD = 3.0
    STEREO_UNIQUENESS_RATIO = 10
    STEREO_SPECKLE_WINDOW = 100
    STEREO_SPECKLE_RANGE = 32
    
    USE_SGBM = True
    WLS_LAMBDA = 8000
    WLS_SIGMA = 1.5
    
    MATCH_CONFIDENCE_THRESHOLD = 0.3
    APPEARANCE_WEIGHT = 0.2
    EPIPOLAR_WEIGHT = 0.4
    DISTANCE_WEIGHT = 0.4
    
    TRACKER_TYPE = "KCF"
    MAX_TRACKERS = 10
    TRACK_MAX_MISSES = 10
    TRACK_MIN_HITS = 3
    
    KALMAN_PROCESS_NOISE = 0.1
    KALMAN_MEASUREMENT_NOISE = 10.0
    
    FPS = 100
    PIXEL_TO_MM = 0.1
    
    SMOOTH_TRAJECTORY = True
    SMOOTH_WINDOW_SIZE = 3
    
    SAVE_TRAJECTORY = True
    TRAJECTORY_OUTPUT_PATH = "output/insect_trajectory_3d.csv"
    VISUALIZE_2D = True
    VISUALIZE_3D = True
    SAVE_VISUALIZATION = True
    
    ENABLE_BEHAVIOR_CLASSIFICATION = True
    SAVE_BEHAVIOR_RESULTS = True
    BEHAVIOR_OUTPUT_PATH = "output/behavior_classification.csv"
    MIN_TRAJECTORY_LENGTH = 10
    FEATURE_SMOOTHING_WINDOW = 5
    
    BEHAVIOR_CLASSIFIER = "HMM"
    USE_LSTM_BEHAVIOR = False
    LSTM_MODEL_PATH = "models/behavior_lstm.pth"
    LSTM_HIDDEN_SIZE = 64
    LSTM_NUM_LAYERS = 2
    
    USE_RAFT_STEREO = False
    RAFT_MODEL_PATH = "models/raft-stereo.pth"
    RAFT_ITERS = 12
    
    LEFT_CAMERA_MATRIX = np.array([
        [1000.0, 0.0, 320.0],
        [0.0, 1000.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    LEFT_DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    RIGHT_CAMERA_MATRIX = np.array([
        [1000.0, 0.0, 320.0],
        [0.0, 1000.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    RIGHT_DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    T = np.array([-50.0, 0.0, 0.0])
    
    Q = None
