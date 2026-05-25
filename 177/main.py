import cv2
import numpy as np
import os
import argparse
from typing import Optional, Dict
from tqdm import tqdm
from config import Config
from camera_calibration import StereoCalibration
from insect_detector import InsectDetector
from insect_tracker import InsectTracker
from stereo_matcher import StereoMatcher
from trajectory_manager import TrajectoryManager
from visualizer import Visualizer
from feature_extractor import FeatureExtractor
from behavior_classifier import BehaviorClassifier, BehaviorClassification


class Insect3DTracker:
    def __init__(self, config: Config):
        self.config = config
        self.calibration = StereoCalibration(config)
        self.detector = InsectDetector(config)
        self.left_tracker = InsectTracker(config)
        self.right_tracker = InsectTracker(config)
        self.stereo_matcher: Optional[StereoMatcher] = None
        self.trajectory_manager = TrajectoryManager(config)
        self.visualizer = Visualizer(config)
        self.feature_extractor = FeatureExtractor(config)
        self.behavior_classifier = BehaviorClassifier(config)
        
        self.left_cap: Optional[cv2.VideoCapture] = None
        self.right_cap: Optional[cv2.VideoCapture] = None
        self.frame_index = 0
        self.behavior_classifications: Dict[int, BehaviorClassification] = {}
        
    def load_or_calibrate(self, calibration_file: Optional[str] = None):
        if calibration_file and os.path.exists(calibration_file):
            print(f"Loading calibration from {calibration_file}...")
            self.calibration.load_calibration(calibration_file)
        else:
            print("Performing stereo calibration...")
            left_pattern = self.config.CALIBRATION_IMAGE_PATTERN.format(cam="left", idx="*")
            right_pattern = self.config.CALIBRATION_IMAGE_PATTERN.format(cam="right", idx="*")
            self.calibration.calibrate_stereo(left_pattern, right_pattern)
            
            if calibration_file:
                os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
                self.calibration.save_calibration(calibration_file)
        
        self.stereo_matcher = StereoMatcher(self.config, self.calibration)
        print("Calibration loaded successfully.")
        
    def open_videos(self, left_path: Optional[str] = None, right_path: Optional[str] = None):
        left_path = left_path or self.config.VIDEO_LEFT_PATH
        right_path = right_path or self.config.VIDEO_RIGHT_PATH
        
        self.left_cap = cv2.VideoCapture(left_path)
        self.right_cap = cv2.VideoCapture(right_path)
        
        if not self.left_cap.isOpened():
            raise ValueError(f"Cannot open left video: {left_path}")
        if not self.right_cap.isOpened():
            raise ValueError(f"Cannot open right video: {right_path}")
        
        print(f"Videos opened successfully.")
        print(f"Left video: {int(self.left_cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
        print(f"Right video: {int(self.right_cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
        
    def process_frame(self) -> Optional[np.ndarray]:
        if self.left_cap is None or self.right_cap is None:
            raise ValueError("Videos not opened. Call open_videos() first.")
        if self.stereo_matcher is None:
            raise ValueError("Calibration not loaded. Call load_or_calibrate() first.")
        
        ret_left, left_frame = self.left_cap.read()
        ret_right, right_frame = self.right_cap.read()
        
        if not ret_left or not ret_right:
            return None
        
        left_rect, right_rect = self.calibration.rectify_images(left_frame, right_frame)
        
        left_detections = self.detector.detect(left_rect)
        right_detections = self.detector.detect(right_rect)
        
        left_tracks = self.left_tracker.update(left_detections)
        right_tracks = self.right_tracker.update(right_detections)
        
        stereo_matches = self.stereo_matcher.match_tracks_by_epipolar(
            left_tracks, right_tracks, epipolar_threshold=5
        )
        
        for track_id, stereo_match in stereo_matches.items():
            self.trajectory_manager.add_stereo_match(stereo_match, self.frame_index)
        
        disparity_map = self.stereo_matcher.compute_disparity_map(left_rect, right_rect)
        
        vis_frame = None
        if self.config.VISUALIZE_2D:
            vis_frame = self.visualizer.create_composite_view(
                left_rect, right_rect, disparity_map,
                left_tracks, right_tracks, stereo_matches
            )
            
            cv2.putText(vis_frame, f"Frame: {self.frame_index}", 
                       (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_frame, f"Tracks: {len(stereo_matches)}", 
                       (200, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.frame_index += 1
        return vis_frame
    
    def run(self, max_frames: Optional[int] = None, display: bool = True):
        print("Starting 3D insect tracking with behavior classification...")
        
        total_frames = int(self.left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        try:
            while True:
                if max_frames and self.frame_index >= max_frames:
                    break
                    
                vis_frame = self.process_frame()
                
                if vis_frame is None:
                    break
                
                if display and self.config.VISUALIZE_2D:
                    cv2.imshow('Insect 3D Tracking', vis_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Processing stopped by user.")
                        break
                    elif key == ord(' '):
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)
                
                pbar.update(1)
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
        finally:
            pbar.close()
            
            if self.config.get('ENABLE_BEHAVIOR_CLASSIFICATION', True):
                self.run_behavior_classification()
            
            self.cleanup()
            
            if self.config.SAVE_TRAJECTORY:
                self.trajectory_manager.save_to_csv()
            
            if self.config.VISUALIZE_3D:
                self.generate_visualizations()
            
            self.print_statistics()
            
            if self.config.get('ENABLE_BEHAVIOR_CLASSIFICATION', True):
                self.print_behavior_report()
    
    def generate_visualizations(self):
        print("\nGenerating visualizations...")
        os.makedirs("output", exist_ok=True)
        
        self.visualizer.plot_3d_trajectory(
            self.trajectory_manager,
            save_path="output/trajectory_3d.png",
            show_plot=False
        )
        
        self.visualizer.plot_trajectory_projections(
            self.trajectory_manager,
            save_path="output/trajectory_projections.png",
            show_plot=False
        )
        
        self.visualizer.plot_trajectory_statistics(
            self.trajectory_manager,
            save_path="output/trajectory_statistics.png",
            show_plot=False
        )
    
    def print_statistics(self):
        print("\n" + "="*60)
        print("TRACKING STATISTICS")
        print("="*60)
        
        stats = self.trajectory_manager.get_trajectory_statistics()
        
        if not stats:
            print("No valid trajectories found.")
            return
        
        print(f"Total frames processed: {self.frame_index}")
        print(f"Number of tracks: {len(stats)}")
        print()
        
        for track_id, track_stats in sorted(stats.items()):
            print(f"Track {track_id}:")
            print(f"  Duration: {track_stats['duration']:.2f}s")
            print(f"  Data points: {track_stats['num_points']}")
            print(f"  Total distance: {track_stats['total_distance']:.2f}mm")
            print(f"  Average speed: {track_stats['avg_speed']:.2f}mm/s")
            print(f"  Maximum speed: {track_stats['max_speed']:.2f}mm/s")
            bb = track_stats['bounding_box']
            print(f"  Bounding box:")
            print(f"    X: [{bb['x_min']:.1f}, {bb['x_max']:.1f}] mm")
            print(f"    Y: [{bb['y_min']:.1f}, {bb['y_max']:.1f}] mm")
            print(f"    Z: [{bb['z_min']:.1f}, {bb['z_max']:.1f}] mm")
            print()
    
    def run_behavior_classification(self):
        print("\n" + "="*60)
        print("Running behavior classification...")
        print("="*60)
        
        trajectories = self.trajectory_manager.get_active_trajectories()
        
        if not trajectories:
            print("No valid trajectories for behavior classification.")
            return
        
        features_list = []
        for trajectory in trajectories:
            features = self.feature_extractor.extract_features(trajectory)
            if features is not None:
                features_list.append(features)
                
                classification = self.behavior_classifier.classify_trajectory(features)
                self.behavior_classifications[trajectory.track_id] = classification
                
                label = self.behavior_classifier.get_behavior_label(classification.dominant_behavior)
                print(f"Track {trajectory.track_id}: {label} (confidence: {max(classification.behavior_distribution.values()):.2%})")
        
        if self.config.get('SAVE_BEHAVIOR_RESULTS', True):
            os.makedirs("output", exist_ok=True)
            self.behavior_classifier.save_classifications("output/behavior_classification.csv")
    
    def print_behavior_report(self):
        if not self.behavior_classifications:
            return
        
        report = self.behavior_classifier.generate_behavior_report()
        print(report)
        
        print("\nBehavior Legend:")
        print("  静止(RESTING)   - 低速度、低加速度、低曲率")
        print("  爬行(WALKING)   - 低速度、中等加速度")
        print("  飞行(FLYING)    - 高速度、低加速度")
        print("  逃逸(ESCAPING)  - 高速度、高加速度、高曲率")
        print("  求偶(COURTING)  - 中等速度、高曲率、高角速度")
        print("  觅食(FORAGING)  - 中等速度、中等曲率")
        print("  攻击(AGGRESSIVE)- 高速度、高加速度、高角速度")
    
    def visualize_behavior_on_frame(self, frame, stereo_matches):
        vis_frame = frame.copy()
        
        for track_id, stereo_match in stereo_matches.items():
            if track_id in self.behavior_classifications:
                classification = self.behavior_classifications[track_id]
                left_point = stereo_match.left_point
                
                vis_frame = self.behavior_classifier.visualize_behavior(
                    vis_frame, classification, left_point
                )
        
        return vis_frame
    
    def cleanup(self):
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="3D Insect Flight Trajectory Tracking")
    parser.add_argument("--left-video", type=str, help="Path to left camera video")
    parser.add_argument("--right-video", type=str, help="Path to right camera video")
    parser.add_argument("--calibration", type=str, help="Path to calibration file (.npz)")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--calibrate", action="store_true", help="Perform calibration")
    args = parser.parse_args()
    
    config = Config()
    
    tracker = Insect3DTracker(config)
    
    if args.calibrate:
        calibration_file = args.calibration or "calibration/stereo_calibration.npz"
        tracker.load_or_calibrate(calibration_file)
        print("Calibration complete.")
        return
    
    calibration_file = args.calibration or "calibration/stereo_calibration.npz"
    tracker.load_or_calibrate(calibration_file)
    
    tracker.open_videos(args.left_video, args.right_video)
    
    tracker.run(
        max_frames=args.max_frames,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
