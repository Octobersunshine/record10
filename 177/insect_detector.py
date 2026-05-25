import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from config import Config


@dataclass
class Detection:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    centroid: Tuple[int, int]


class InsectDetector:
    def __init__(self, config: Config):
        self.config = config
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.BACKGROUND_SUBTRACTOR_HISTORY,
            varThreshold=16,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None:
            return []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        fg_mask = self.bg_subtractor.apply(gray)
        
        _, fg_mask = cv2.threshold(fg_mask, self.config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.config.INSECT_MIN_AREA <= area <= self.config.INSECT_MAX_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = x + w // 2
                    cy = y + h // 2
                
                confidence = min(1.0, area / self.config.INSECT_MAX_AREA)
                
                detections.append(Detection(
                    x=x, y=y, width=w, height=h,
                    confidence=confidence,
                    centroid=(cx, cy)
                ))
        
        return detections
    
    def detect_from_rectified(self, left_rect: np.ndarray, right_rect: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        left_detections = self.detect(left_rect)
        right_detections = self.detect(right_rect)
        return left_detections, right_detections
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Detection], 
                            color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        vis_frame = frame.copy()
        for det in detections:
            cv2.rectangle(vis_frame, (det.x, det.y), 
                         (det.x + det.width, det.y + det.height), color, 2)
            cv2.circle(vis_frame, det.centroid, 3, (0, 0, 255), -1)
            cv2.putText(vis_frame, f"{det.confidence:.2f}", 
                       (det.x, det.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis_frame
