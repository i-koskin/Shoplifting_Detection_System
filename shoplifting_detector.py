"""
Real-time Shoplifting Detection System

This system uses YOLO for pose detection and XGBoost for behavior classification
to identify suspicious activities in real-time video streams.

Features:
- Pose detection and tracking
- Behavior classification using machine learning
- Evidence saving (images and video clips)
- Configurable validation thresholds for suspicious behavior
"""

import cv2
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone
import torch
import logging
import os
import time
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from dotenv import load_dotenv
import json


# Creating a logger for the current module
logger = logging.getLogger(__name__)


def load_environment():
    """
    Load environment variables with priority for environment-specific files.

    Looks for .env.{environment} file based on MONITORING_ENV variable.
    Falls back to system environment variables if file not found.
    """
    env = os.getenv('MONITORING_ENV', 'test')
    env_file = f'.env.{env}'

    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        logger.warning(
            f"[CONFIG] Environment file {env_file} not found, using environment variables only")


# Load environment variables early in the process
load_environment()


class Config:
    """
    Configuration manager for the shoplifting detection system.

    Handles all configuration parameters from environment variables with
    environment-specific overrides for test and production modes.
    """

    def __init__(self):
        """Initialize configuration with default values and environment overrides."""

        # Basic configuration
        self.MONITORING_ENV = os.getenv('MONITORING_ENV', 'test')
        self.CAMERA_NAME = os.getenv('CAMERA_NAME', 'cam1')

        # Determine VIDEO_SOURCE depending on the operating mode and the camera used
        if self.MONITORING_ENV == 'test':
            # In test mode, use video files from test_videos folder
            self.camera_number = self.CAMERA_NAME.replace(
                'cam', '') if 'cam' in self.CAMERA_NAME else '1'
            self.VIDEO_SOURCE = os.getenv(
                'VIDEO_SOURCE', f'./test_videos/test_{self.CAMERA_NAME}.mp4')
        else:
            # In production mode, use RTSP stream from environment variables
            self.VIDEO_SOURCE = os.getenv(
                'VIDEO_SOURCE', f'rtsp://camera_{self.CAMERA_NAME}_url')

        # AI Models
        self.YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolo11s-pose.pt')
        available_model = f'trained_model_{self.camera_number}.json'
        if available_model:
            self.XGBOOST_MODEL_PATH = available_model
            logger.info(
                f"Camera {self.CAMERA_NAME} using XGBoost model: {self.XGBOOST_MODEL_PATH}")
        else:
            raise ValueError("XGBoost model not loaded")

        # Detection settings
        self.CUDA_AVAILABLE = os.getenv(
            'CUDA_AVAILABLE', 'true').lower() == 'true'
        self.DETECTION_CONFIDENCE = float(
            os.getenv('DETECTION_CONFIDENCE', '0.5'))

        # Alert settings
        self.ALERT_COOLDOWN = int(os.getenv('ALERT_COOLDOWN', '10'))
        self.MAX_ALERTS_PER_MINUTE = int(
            os.getenv('MAX_ALERTS_PER_MINUTE', '5'))
        self.ALERT_THRESHOLD = int(os.getenv('ALERT_THRESHOLD', '10'))

        # System behavior
        self.SAVE_EVIDENCE = os.getenv(
            'SAVE_EVIDENCE', 'true').lower() == 'true'
        self.SHOW_DISPLAY = os.getenv(
            'SHOW_DISPLAY', 'false').lower() == 'true'
        self.LOG_DETAILED_PREDICTIONS = os.getenv(
            'LOG_DETAILED_PREDICTIONS', 'true').lower() == 'true'
        self.FRAME_SKIP = int(os.getenv('FRAME_SKIP', '1'))

        # Logging
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv(
            'LOG_FILE', f'./logs/shoplifting_{self.CAMERA_NAME}.log')

        # Paths
        self.EVIDENCE_IMAGE_PATH = os.getenv(
            'EVIDENCE_IMAGE_PATH', f'./alerts/{self.CAMERA_NAME}/alerts_images')
        self.EVIDENCE_VIDEO_PATH = os.getenv(
            'EVIDENCE_VIDEO_PATH', f'./alerts/{self.CAMERA_NAME}/alerts_videos')
        self.DEBUG_PATH = os.getenv(
            'DEBUG_PATH', f'./debug/{self.CAMERA_NAME}')

        # Performance
        self.FRAME_BUFFER_SIZE = int(
            os.getenv('FRAME_BUFFER_SIZE', '450'))  # 15 seconds at 30 FPS
        self.OUTPUT_FPS = float(os.getenv('OUTPUT_FPS', '30.0'))
        self.VIDEO_CODEC = os.getenv('VIDEO_CODEC', 'mp4v')

        # Display
        self.DISPLAY_WIDTH = int(os.getenv('DISPLAY_WIDTH', '1020'))
        self.DISPLAY_HEIGHT = int(os.getenv('DISPLAY_HEIGHT', '600'))
        self.WINDOW_NAME = os.getenv(
            'WINDOW_NAME', f'Shoplifting Detection - {self.CAMERA_NAME}')

        # State transition settings
        self.NORMAL_STATE_THRESHOLD = int(
            os.getenv('NORMAL_STATE_THRESHOLD', '10'))
        self.MIN_NORMAL_DURATION = float(
            os.getenv('MIN_NORMAL_DURATION', '1.0'))

        # Suspicious behavior validation settings
        self.SUSPICIOUS_CONSECUTIVE_THRESHOLD = int(
            os.getenv('SUSPICIOUS_CONSECUTIVE_THRESHOLD', '5'))
        self.SUSPICIOUS_CONFIDENCE_THRESHOLD = float(
            os.getenv('SUSPICIOUS_CONFIDENCE_THRESHOLD', '0.3'))
        self.MIN_SUSPICIOUS_DURATION = float(
            os.getenv('MIN_SUSPICIOUS_DURATION', '1.0'))

        # Apply environment-specific configuration overrides
        self._apply_environment_overrides()

        # Log configuration
        logger.info(f"[CONFIG] VALIDATION SETTINGS - "
                    f"CONSECUTIVE_THRESHOLD: {self.SUSPICIOUS_CONSECUTIVE_THRESHOLD}, "
                    f"MIN_DURATION: {self.MIN_SUSPICIOUS_DURATION}, "
                    f"CONFIDENCE_THRESHOLD: {self.SUSPICIOUS_CONFIDENCE_THRESHOLD}")

    def _apply_environment_overrides(self):
        """
        Apply environment-specific configuration overrides.

        Different settings are applied for production vs test environments
        to optimize performance and behavior for each use case.
        """
        if self.MONITORING_ENV == 'production':
            # Production settings - optimized for reliability and performance
            self.MIN_CONFIDENCE = float(
                os.getenv('PRODUCTION_MIN_CONFIDENCE', '0.7'))
            self.ALERT_THRESHOLD = int(
                os.getenv('PRODUCTION_ALERT_THRESHOLD', '10'))
            self.FRAME_SKIP = int(os.getenv('PRODUCTION_FRAME_SKIP', '2'))
            self.SHOW_DISPLAY = os.getenv(
                'PRODUCTION_SHOW_DISPLAY', 'false').lower() == 'true'
            self.LOG_DETAILED_PREDICTIONS = os.getenv(
                'PRODUCTION_LOG_DETAILED_PREDICTIONS', 'false').lower() == 'true'
            self.LOG_LEVEL = os.getenv('PRODUCTION_LOG_LEVEL', 'INFO')
            self.NORMAL_STATE_THRESHOLD = int(
                os.getenv('PRODUCTION_NORMAL_STATE_THRESHOLD', '10'))
            self.MIN_NORMAL_DURATION = float(
                os.getenv('PRODUCTION_MIN_NORMAL_DURATION', '1.0'))

            # Production validation settings - stricter thresholds
            self.SUSPICIOUS_CONSECUTIVE_THRESHOLD = int(
                os.getenv('PRODUCTION_SUSPICIOUS_CONSECUTIVE_THRESHOLD', '10'))
            self.SUSPICIOUS_CONFIDENCE_THRESHOLD = float(
                os.getenv('PRODUCTION_SUSPICIOUS_CONFIDENCE_THRESHOLD', '0.3'))
            self.MIN_SUSPICIOUS_DURATION = float(
                os.getenv('PRODUCTION_MIN_SUSPICIOUS_DURATION', '1.0'))

        elif self.MONITORING_ENV == 'test':
            # Test settings - optimized for testing
            self.MIN_CONFIDENCE = float(
                os.getenv('TEST_MIN_CONFIDENCE', '0.7'))
            self.ALERT_THRESHOLD = int(os.getenv('TEST_ALERT_THRESHOLD', '5'))
            self.FRAME_SKIP = int(os.getenv('TEST_FRAME_SKIP', '1'))
            self.SHOW_DISPLAY = os.getenv(
                'TEST_SHOW_DISPLAY', 'true').lower() == 'true'
            self.LOG_DETAILED_PREDICTIONS = os.getenv(
                'TEST_LOG_DETAILED_PREDICTIONS', 'true').lower() == 'true'
            self.LOG_LEVEL = os.getenv('TEST_LOG_LEVEL', 'DEBUG')
            self.NORMAL_STATE_THRESHOLD = int(
                os.getenv('TEST_NORMAL_STATE_THRESHOLD', '10'))
            self.MIN_NORMAL_DURATION = float(
                os.getenv('TEST_MIN_NORMAL_DURATION', '1.0'))

            # Test validation settings
            self.SUSPICIOUS_CONSECUTIVE_THRESHOLD = int(
                os.getenv('TEST_SUSPICIOUS_CONSECUTIVE_THRESHOLD', '10'))
            self.SUSPICIOUS_CONFIDENCE_THRESHOLD = float(
                os.getenv('TEST_SUSPICIOUS_CONFIDENCE_THRESHOLD', '0.3'))
            self.MIN_SUSPICIOUS_DURATION = float(
                os.getenv('TEST_MIN_SUSPICIOUS_DURATION', '1.0'))

        else:
            # Fallback for unknown modes - use production settings
            logger.warning(
                f"Unknown monitoring environment: {self.MONITORING_ENV}, using production settings")
            self.MIN_CONFIDENCE = float(
                os.getenv('PRODUCTION_MIN_CONFIDENCE', '0.7'))
            self.ALERT_THRESHOLD = int(
                os.getenv('PRODUCTION_ALERT_THRESHOLD', '10'))
            self.FRAME_SKIP = int(os.getenv('PRODUCTION_FRAME_SKIP', '2'))
            self.SHOW_DISPLAY = os.getenv(
                'PRODUCTION_SHOW_DISPLAY', 'false').lower() == 'true'
            self.LOG_DETAILED_PREDICTIONS = os.getenv(
                'PRODUCTION_LOG_DETAILED_PREDICTIONS', 'false').lower() == 'true'
            self.LOG_LEVEL = os.getenv('PRODUCTION_LOG_LEVEL', 'INFO')
            self.NORMAL_STATE_THRESHOLD = int(
                os.getenv('PRODUCTION_NORMAL_STATE_THRESHOLD', '10'))
            self.MIN_NORMAL_DURATION = float(
                os.getenv('PRODUCTION_MIN_NORMAL_DURATION', '1.0'))

            # Validation settings fallback
            self.SUSPICIOUS_CONSECUTIVE_THRESHOLD = int(
                os.getenv('PRODUCTION_SUSPICIOUS_CONSECUTIVE_THRESHOLD', '10'))
            self.SUSPICIOUS_CONFIDENCE_THRESHOLD = float(
                os.getenv('PRODUCTION_SUSPICIOUS_CONFIDENCE_THRESHOLD', '0.3'))
            self.MIN_SUSPICIOUS_DURATION = float(
                os.getenv('PRODUCTION_MIN_SUSPICIOUS_DURATION', '1.0'))


# Initialize global configuration
config = Config()

# Create necessary directories
Path('./logs').mkdir(parents=True, exist_ok=True)
Path(config.EVIDENCE_IMAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(config.EVIDENCE_VIDEO_PATH).mkdir(parents=True, exist_ok=True)
Path(config.DEBUG_PATH).mkdir(parents=True, exist_ok=True)
Path('./test_videos').mkdir(parents=True, exist_ok=True)


# Configure logging system
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(environment)s] - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class ContextFilter(logging.Filter):
    """Logging filter to add environment context to log records."""

    def filter(self, record):
        """Add environment information to log record."""
        record.environment = config.MONITORING_ENV
        return True


logger.addFilter(ContextFilter())


class AlertManager:
    """
    Manages alert sending with cooldown periods and rate limiting.

    Prevents alert spam by enforcing cooldowns per track and overall rate limits.
    In test mode, alerts are stored locally instead of being sent via API.
    """

    def __init__(self):
        """Initialize alert manager with cooldown and rate limiting settings."""
        self.cooldown = config.ALERT_COOLDOWN
        self.max_per_minute = config.MAX_ALERTS_PER_MINUTE
        self.last_alert_time = {}  # Track-specific cooldown
        self.alert_history = deque()  # Global rate limiting
        self.test_alerts = []  # Store test alerts for analysis

    def can_send_alert(self, track_id):
        """
        Check if an alert can be sent for the given track ID.

        Args:
            track_id (int): The track ID to check cooldown for

        Returns:
            bool: True if alert can be sent, False otherwise
        """
        current_time = time.time()

        # Check cooldown for specific track_id
        if track_id in self.last_alert_time:
            if current_time - self.last_alert_time[track_id] < self.cooldown:
                return False

        # Check overall rate limit
        minute_ago = current_time - 60
        while self.alert_history and self.alert_history[0] < minute_ago:
            self.alert_history.popleft()

        if len(self.alert_history) >= self.max_per_minute:
            logger.warning(
                f"Rate limit exceeded: {len(self.alert_history)} alerts in last minute")
            return False

        return True

    def record_alert(self, track_id):
        """
        Record that an alert was sent for a track ID.

        Args:
            track_id (int): The track ID that triggered the alert
        """
        current_time = time.time()
        self.last_alert_time[track_id] = current_time
        self.alert_history.append(current_time)

    def send_test_alert(self, track_id, label, data):
        """
        Save test alert without sending via API.

        In test mode, alerts are stored locally for analysis and debugging.

        Args:
            track_id (int): The track ID that triggered the alert
            label (str): The alert label (e.g., "Suspicious")
            data (dict): Additional alert data for analysis
        """
        alert_data = {
            "track_id": track_id,
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "camera_name": config.CAMERA_NAME
        }

        self.test_alerts.append(alert_data)
        # Save to file for analysis
        if config.MONITORING_ENV == 'test':
            alert_file = f"{config.DEBUG_PATH}/test_alerts_{datetime.now().strftime('%Y-%m-%d')}.json"
            try:
                with open(alert_file, 'a') as f:
                    f.write(json.dumps(alert_data, default=str) + '\n')
            except Exception as e:
                logger.error(f"Error saving test alert: {str(e)}")

        logger.info(f"TEST ALERT - Would send: {alert_data}")


class SuspiciousSequence:
    """
    Represents a sequence of suspicious frames for a single track.

    Tracks consecutive suspicious detections and validates them against
    configurable thresholds for count, duration, and confidence.
    """

    def __init__(self, track_id):
        """
        Initialize a suspicious sequence for a track.

        Args:
            track_id (int): The track ID this sequence belongs to
        """
        self.track_id = track_id
        self.consecutive_count = 0
        self.max_consecutive = 0
        self.start_time = None
        self.last_update = None
        self.confidence_history = []
        self.is_currently_suspicious = False
        self._was_validated = False  # Track if sequence was already validated

    def update(self, is_suspicious, confidence, timestamp):
        """
        Update the sequence state with new detection results.

        Args:
            is_suspicious (bool): Whether current frame is suspicious
            confidence (float): Model confidence score (0-1, lower is more suspicious)
            timestamp (float): Current timestamp

        Returns:
            dict: Current sequence status
        """
        self.last_update = timestamp

        # Save previous state for debugging
        prev_count = self.consecutive_count
        prev_state = self.is_currently_suspicious

        if is_suspicious:
            if confidence <= config.SUSPICIOUS_CONFIDENCE_THRESHOLD:
                # Suspicious behavior with sufficient confidence
                if not self.is_currently_suspicious:
                    # Start new sequence
                    self.start_time = timestamp
                    self.consecutive_count = 1
                    self.is_currently_suspicious = True
                    if config.LOG_DETAILED_PREDICTIONS:
                        logger.debug(
                            f"[NEW] NEW SEQUENCE - Track:{self.track_id}, Confidence:{confidence:.3f}, Count:1")
                else:
                    # Continue existing sequence
                    self.consecutive_count += 1
                    if config.LOG_DETAILED_PREDICTIONS:
                        logger.debug(
                            f"[CONTINUE] CONTINUE SEQUENCE - Track:{self.track_id}, Confidence:{confidence:.3f}, Count:{self.consecutive_count}")

                self.max_consecutive = max(
                    self.max_consecutive, self.consecutive_count)

                # Store confidence for validation (with size limit for memory efficiency)
                if len(self.confidence_history) < config.SUSPICIOUS_CONSECUTIVE_THRESHOLD * 2:
                    self.confidence_history.append(confidence)
            else:
                # Suspicious behavior but low confidence - maintain sequence but don't increment
                self.is_currently_suspicious = False
                self.consecutive_count = 0
                if config.LOG_DETAILED_PREDICTIONS:
                    logger.debug(
                        f"[WARNING] LOW CONFIDENCE START - Track:{self.track_id}, Confidence:{confidence:.3f} < {config.SUSPICIOUS_CONFIDENCE_THRESHOLD}")
        else:
            # Normal behavior - interrupt sequence
            if self.is_currently_suspicious and config.LOG_DETAILED_PREDICTIONS:
                logger.debug(
                    f"[STOP] SEQUENCE INTERRUPTED - Track:{self.track_id}, was:{prev_count}, now:0")
            self.is_currently_suspicious = False
            self.consecutive_count = 0

        # Log state changes
        if config.LOG_DETAILED_PREDICTIONS and (prev_count != self.consecutive_count or prev_state != self.is_currently_suspicious):
            logger.debug(
                f"[TRANSITION] SEQUENCE UPDATE - Track:{self.track_id}, Count:{prev_count}->{self.consecutive_count}, State:{prev_state}->{self.is_currently_suspicious}")

        return self.get_status()

    def get_status(self):
        """
        Get current sequence status.

        Returns:
            dict: Current sequence status including count, duration, and confidence
        """
        duration = 0
        if self.start_time and self.last_update:
            duration = self.last_update - self.start_time

        avg_confidence = np.mean(
            self.confidence_history) if self.confidence_history else 0

        return {
            'track_id': self.track_id,
            'consecutive_count': self.consecutive_count,
            'max_consecutive': self.max_consecutive,
            'duration': duration,
            'avg_confidence': avg_confidence,
            'is_active': self.is_currently_suspicious,
            'is_valid': self.is_valid()
        }

    def is_valid(self):
        """
        Check if the sequence meets validation criteria.

        Returns:
            bool: True if sequence is valid (meets all thresholds)
        """
        if not self.is_currently_suspicious:
            return False

        # For test mode, use simplified validation (duration check disabled)
        if config.MONITORING_ENV == 'test':
            count_ok = self.consecutive_count >= config.SUSPICIOUS_CONSECUTIVE_THRESHOLD
            if count_ok and not self._was_validated:
                logger.debug(
                    f"[TEST VALIDATION] Sequence validated - Track:{self.track_id}, Count:{self.consecutive_count}")
                self._was_validated = True
            return count_ok

        duration_ok = True
        duration = 0
        if self.start_time and self.last_update:
            duration = self.last_update - self.start_time
            duration_ok = duration >= config.MIN_SUSPICIOUS_DURATION

        count_ok = self.consecutive_count >= config.SUSPICIOUS_CONSECUTIVE_THRESHOLD

        confidence_ok = True
        avg_confidence = 0
        if self.confidence_history:
            avg_confidence = np.mean(self.confidence_history)
            confidence_ok = avg_confidence <= config.SUSPICIOUS_CONFIDENCE_THRESHOLD

        is_valid_result = duration_ok and count_ok and confidence_ok

        # Debug logging for validation checks
        if config.LOG_DETAILED_PREDICTIONS and self.consecutive_count >= config.SUSPICIOUS_CONSECUTIVE_THRESHOLD:
            logger.debug(f"[CHECK] VALIDATION CHECK - Track:{self.track_id}, "
                         f"Count:{self.consecutive_count}(need:{config.SUSPICIOUS_CONSECUTIVE_THRESHOLD})={count_ok}, "
                         f"Duration:{duration:.2f}s(need:{config.MIN_SUSPICIOUS_DURATION})={duration_ok}, "
                         f"Conf:{avg_confidence:.3f}(need:{config.SUSPICIOUS_CONFIDENCE_THRESHOLD})={confidence_ok}")

        return is_valid_result


class SuspiciousSequenceTracker:
    """
    Tracks suspicious sequences across multiple tracks.

    Manages SuspiciousSequence objects for each track and provides
    methods to validate, cleanup, and monitor sequences.
    """

    def __init__(self):
        """Initialize sequence tracker with empty sequences dictionary."""
        self.sequences = {}  # {track_id: SuspiciousSequence}

    def update(self, track_id, is_suspicious, confidence, timestamp):
        """
        Update sequence state for a track.

        Args:
            track_id (int): Track ID to update
            is_suspicious (bool): Whether current detection is suspicious
            confidence (float): Model confidence score
            timestamp (float): Current timestamp

        Returns:
            dict: Updated sequence status
        """
        if track_id not in self.sequences:
            self.sequences[track_id] = SuspiciousSequence(track_id)

        sequence = self.sequences[track_id]
        return sequence.update(is_suspicious, confidence, timestamp)

    def get_valid_sequences(self):
        """
        Get all valid sequences that meet validation criteria.

        Returns:
            dict: Dictionary of valid sequences {track_id: sequence}
        """
        return {track_id: seq for track_id, seq in self.sequences.items() if seq.is_valid()}

    def get_active_suspicious_count(self):
        """
        Get count of currently active suspicious sequences.

        Returns:
            int: Number of active suspicious sequences
        """
        return sum(1 for seq in self.sequences.values() if seq.is_currently_suspicious)

    def cleanup_inactive(self, active_track_ids):
        """
        Remove sequences for tracks that are no longer active.

        Args:
            active_track_ids (set): Set of currently active track IDs

        Returns:
            int: Number of sequences removed
        """
        inactive_tracks = set(self.sequences.keys()) - set(active_track_ids)
        removed_count = len(inactive_tracks)
        for track_id in inactive_tracks:
            del self.sequences[track_id]
        return removed_count

    def reset_all_sequences(self):
        """
        Reset all sequences (when no tracks are detected).

        Returns:
            int: Number of sequences reset
        """
        removed_count = len(self.sequences)
        self.sequences.clear()
        return removed_count

    def reset_sequence(self, track_id):
        """
        Reset sequence for a specific track.

        Args:
            track_id (int): Track ID to reset
        """
        if track_id in self.sequences:
            del self.sequences[track_id]


class ShopliftingDetector:
    """
    Main shoplifting detection class.

    Coordinates YOLO pose detection, XGBoost classification, alert management,
    and evidence saving to detect suspicious behaviors in real-time.
    """

    def __init__(self):
        """Initialize detector with models, trackers, and configuration."""
        self.device = 'cuda' if torch.cuda.is_available() and config.CUDA_AVAILABLE else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Load AI models
        try:
            self.model_yolo = YOLO(config.YOLO_MODEL_PATH)
            self.model_xgb = xgb.Booster()
            self.model_xgb.load_model(config.XGBOOST_MODEL_PATH)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

        # Initialize components
        self.alert_manager = AlertManager()
        self.suspicious_count = defaultdict(int)
        self.frame_buffer = deque(
            maxlen=config.FRAME_BUFFER_SIZE)  # 15 seconds buffer
        self.frame_count = 0
        self.detection_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'alerts': 0
        }

        # Storage for recent detections
        self.last_detections = defaultdict(list)
        self.max_detections_per_track = 10

        # Track suspicious states and transitions
        self.first_suspicious_time = defaultdict(float)
        self.first_suspicious_frame = {}
        self.image_saved_for_track = set()
        self.video_saved_for_track = set()
        self.current_state_was_suspicious = {}
        self.last_state_was_suspicious = {}
        self.normal_start_time = {}  # Start time of normal state for transition timing
        self.normal_duration = {}    # Duration of normal state for transition validation

        # Sequence validation system
        self.sequence_tracker = SuspiciousSequenceTracker()
        self.validated_tracks = set()  # Tracks that passed validation
        self.validation_stats = {
            'total_sequences': 0,
            'valid_sequences': 0,
            'interrupted_sequences': 0
        }

    def save_evidence(self, track_id, label, prediction_score, is_first_event=False, is_transition=False, is_disappearance=False):
        """
        Save image and video evidence for alerts.

        Only saves evidence for "Suspicious" labeled events.

        Args:
            track_id (int): Track ID for the evidence
            label (str): Behavior label ("Suspicious" or "Normal")
            prediction_score (float): Model prediction score
            is_first_event (bool): Whether this is the first suspicious event
            is_transition (bool): Whether this is a transition event
            is_disappearance (bool): Whether track disappeared while suspicious
        """
        # Only save evidence for "Suspicious" events
        if label != "Suspicious":
            if config.LOG_DETAILED_PREDICTIONS:
                logger.debug(
                    f"Evidence saving skipped - label '{label}' is not 'Suspicious'")
            return

        if not config.SAVE_EVIDENCE:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        try:
            # Save image only for the first suspicious event
            if is_first_event and track_id not in self.image_saved_for_track:
                if track_id in self.first_suspicious_frame:
                    img_filename = f"{config.EVIDENCE_IMAGE_PATH}/{timestamp}_{track_id}_{label}.jpg"
                    success = cv2.imwrite(
                        img_filename, self.first_suspicious_frame[track_id])
                    if success:
                        logger.info(
                            f"Saved first event image evidence: {img_filename}")
                        self.image_saved_for_track.add(track_id)
                    else:
                        logger.error(
                            f"Failed to save image evidence: {img_filename}")

            # Save video clip for transition from Suspicious to Normal
            if is_transition and track_id not in self.video_saved_for_track:
                self._save_video_evidence(
                    track_id, label, timestamp, "transition")
                self.video_saved_for_track.add(track_id)

            # Save video clip when track disappears while in suspicious state
            if is_disappearance and track_id not in self.video_saved_for_track:
                self._save_video_evidence(
                    track_id, label, timestamp, "disappearance")
                self.video_saved_for_track.add(track_id)

        except Exception as e:
            logger.error(f"Error saving evidence: {str(e)}")

    def _save_video_evidence(self, track_id, label, timestamp, event_type):
        """
        Save video evidence from frame buffer.

        Args:
            track_id (int): Track ID for the evidence
            label (str): Behavior label
            timestamp (str): Timestamp for filename
            event_type (str): Type of event ("transition" or "disappearance")
        """
        # Calculate actual video duration based on buffer size and FPS
        buffer_duration = len(self.frame_buffer) / config.OUTPUT_FPS

        # Define duration thresholds
        min_duration = 5.0  # Minimum 5 seconds
        target_duration = 15.0  # Target 15 seconds

        actual_duration = min(buffer_duration, target_duration)

        if buffer_duration >= min_duration:
            # Determine how many frames to save
            frames_to_save = min(
                len(self.frame_buffer),
                int(target_duration * config.OUTPUT_FPS)
            )

            # Take the most recent frames
            start_index = len(self.frame_buffer) - frames_to_save
            frames_for_video = list(self.frame_buffer)[start_index:]

            if frames_for_video:
                video_filename = f"{config.EVIDENCE_VIDEO_PATH}/{timestamp}_{track_id}_{label}_{event_type}_{int(actual_duration)}s.mp4"
                height, width = frames_for_video[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
                out = cv2.VideoWriter(
                    video_filename, fourcc, config.OUTPUT_FPS, (width, height))

                # Write selected frames
                for video_frame in frames_for_video:
                    out.write(video_frame)

                out.release()

                logger.info(
                    f"Saved {actual_duration:.1f}-second {event_type} video evidence: {video_filename} "
                    f"(used {frames_to_save}/{len(self.frame_buffer)} frames)")
        else:
            logger.warning(
                f"Insufficient video buffer for {event_type} of track_id {track_id}: "
                f"{buffer_duration:.1f}s < {min_duration}s minimum")

    def _handle_validated_suspicious(self, track_id, sequence_status, prediction, frame, current_time):
        """Processes validated suspicious behavior"""
        logger.info(f"[VALIDATED] SUSPICIOUS BEHAVIOR VALIDATED - Track:{track_id}, "
                    f"Consecutive:{sequence_status['consecutive_count']}, "
                    f"Duration:{sequence_status['duration']:.2f}s, "
                    f"AvgConf:{sequence_status['avg_confidence']:.3f}")

        # Add to validated tracks
        self.validated_tracks.add(track_id)

        # Saving the first image
        if track_id not in self.first_suspicious_frame:
            self.first_suspicious_frame[track_id] = frame.copy()
            self.first_suspicious_time[track_id] = current_time

            if track_id not in self.image_saved_for_track:
                self.save_evidence(track_id, "Suspicious",
                                   prediction, is_first_event=True)

        # Updating statistics
        self.validation_stats['valid_sequences'] += 1
        self.suspicious_count[track_id] += 1

        # Checking the threshold for additional alerts
        if self.suspicious_count[track_id] <= config.ALERT_THRESHOLD:
            self.suspicious_count[track_id] = 0

    def _determine_display_properties(self, track_id, is_suspicious, sequence_status, prediction):
        """
        Determine display properties (label and color) based on sequence status.

        Args:
            track_id (int): Track ID
            is_suspicious (bool): Whether current detection is suspicious
            sequence_status (dict): Sequence validation status
            prediction (float): Model prediction score

        Returns:
            tuple: (label, color) for display
        """
        current_time = time.time()
        current_state_was_suspicious = self.last_state_was_suspicious.get(
            track_id, False)

        # Enhanced debug logging for state transitions
        # if config.LOG_DETAILED_PREDICTIONS and sequence_status.get('is_valid', False):
        #     logger.debug(f"[DEBUG VALIDATION] Track:{track_id}:")
        #     logger.debug(f"  - is_suspicious: {is_suspicious}")
        #     logger.debug(
        #         f"  - current_state_was_suspicious: {current_state_was_suspicious}")
        #     logger.debug(
        #         f"  - in validated_tracks: {track_id in self.validated_tracks}")
        #     logger.debug(f"  - sequence_valid: {sequence_status['is_valid']}")

        # Debug logging of transitions
        if config.LOG_DETAILED_PREDICTIONS:
            state_change = ""
            if not current_state_was_suspicious and is_suspicious:
                state_change = "Normal  -> Suspicious"
            elif current_state_was_suspicious and not is_suspicious:
                state_change = "Suspicious  -> Normal"
            elif current_state_was_suspicious and is_suspicious and sequence_status['is_valid']:
                state_change = "Suspicious  -> Validated Suspicious"

            if state_change:
                logger.debug(f"Track {track_id}: {state_change}, "
                             f"Consecutive: {sequence_status['consecutive_count']}, "
                             f"Valid: {sequence_status['is_valid']}")

        if is_suspicious:
            # ALWAYS reset normal state counters when becoming suspicious
            was_timer_active = track_id in self.normal_start_time
            if track_id in self.normal_start_time:
                del self.normal_start_time[track_id]
            if track_id in self.normal_duration:
                del self.normal_duration[track_id]

            if was_timer_active and config.LOG_DETAILED_PREDICTIONS:
                logger.debug(
                    f"[NORMAL TIMER] Reset normal timer for Track:{track_id} - becoming suspicious")

            if sequence_status['is_valid']:
                # VALIDATED SUSPICIOUS STATE
                label = "Suspicious"
                color = (0, 0, 255)  # Red - confirmed

                # SIMPLIFIED transition condition - only check if not already validated
                if (not current_state_was_suspicious
                        and track_id not in self.validated_tracks):
                    logger.info(f"[VALIDATED] Transition to VALIDATED Suspicious - Track:{track_id}, "
                                f"Consecutive:{sequence_status['consecutive_count']}, "
                                f"Duration:{sequence_status['duration']:.2f}s")
                    self.validated_tracks.add(track_id)

                    # Save first image on validation
                    if track_id not in self.image_saved_for_track:
                        self.save_evidence(
                            track_id, "Suspicious", prediction, is_first_event=True)

                self.last_state_was_suspicious[track_id] = True

            else:
                # UNVALIDATED SUSPICIOUS STATE
                progress = f"{sequence_status['consecutive_count']}/{config.SUSPICIOUS_CONSECUTIVE_THRESHOLD}"
                label = f"Suspicious[{progress}]"
                color = (0, 165, 255)  # Orange - in progress
                self.last_state_was_suspicious[track_id] = True

        else:
            # NORMAL STATE
            label = "Normal"
            color = (0, 255, 0)  # Green

            # START normal state timer ONLY if previous state was suspicious
            if (current_state_was_suspicious and
                    track_id not in self.normal_start_time):
                self.normal_start_time[track_id] = current_time
                if config.LOG_DETAILED_PREDICTIONS:
                    logger.debug(
                        f"[NORMAL TIMER] Start normal timer for Track:{track_id}")

            # UPDATE timer ONLY if it was started
            if track_id in self.normal_start_time:
                self.normal_duration[track_id] = current_time - \
                    self.normal_start_time[track_id]

            # Check transition only after confirmed stable normal state
            if (not current_state_was_suspicious and
                track_id in self.validated_tracks and
                track_id in self.normal_duration and
                self.normal_duration[track_id] >= config.MIN_NORMAL_DURATION and
                    track_id not in self.video_saved_for_track):

                logger.info(
                    f"[TRANSITION] Stable transition from VALIDATED Suspicious to Normal - Track:{track_id}, NormalDuration:{self.normal_duration[track_id]:.2f}s")
                self.save_evidence(track_id, "Suspicious",
                                   prediction, is_transition=True)

                # Reset validation and counters
                self.sequence_tracker.reset_sequence(track_id)
                if track_id in self.validated_tracks:
                    self.validated_tracks.remove(track_id)
                if track_id in self.normal_start_time:
                    del self.normal_start_time[track_id]
                if track_id in self.normal_duration:
                    del self.normal_duration[track_id]

            self.last_state_was_suspicious[track_id] = False

        return label, color

    def log_validation_statistics(self):
        """Log sequence validation statistics for monitoring."""
        total_sequences = len(self.sequence_tracker.sequences)
        valid_sequences = len(self.sequence_tracker.get_valid_sequences())
        active_tracks = len(self.validated_tracks)
        active_suspicious = self.sequence_tracker.get_active_suspicious_count()

        avg_consecutive = 0
        if self.sequence_tracker.sequences:
            avg_consecutive = np.mean(
                [seq.consecutive_count for seq in self.sequence_tracker.sequences.values()])

        logger.info(f"[STATS] SEQUENCE VALIDATION - Total:{total_sequences}, "
                    f"ActiveSuspicious:{active_suspicious}, "
                    f"Valid:{valid_sequences}, Confirmed:{active_tracks}, "
                    f"AvgConsecutive:{avg_consecutive:.1f}")

    def monitor_sequence_tracker(self):
        """Monitor and log sequence tracker state for debugging."""
        active_sequences = len(self.sequence_tracker.sequences)
        if active_sequences > 0 and config.LOG_DETAILED_PREDICTIONS:
            for track_id, sequence in self.sequence_tracker.sequences.items():
                logger.debug(f"[STATS] SEQUENCE MONITOR - Track:{track_id}, "
                             f"Count:{sequence.consecutive_count}, "
                             f"Active:{sequence.is_currently_suspicious}")

    def process_frame(self, frame):
        """
        Process a single video frame with sequence-based validation.

        Args:
            frame: Input video frame

        Returns:
            Processed frame with annotations
        """
        self.frame_count += 1
        self.detection_stats['total_frames'] += 1

        # Skip frames for performance
        if self.frame_count % config.FRAME_SKIP != 0:
            return frame

        self.detection_stats['processed_frames'] += 1

        # Add frame to buffer for video evidence
        if config.SAVE_EVIDENCE:
            self.frame_buffer.append(frame.copy())

        # Resize frame for processing
        frame_resized = cv2.resize(
            frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

        try:
            # YOLO pose detection and tracking
            results = self.model_yolo.track(
                frame_resized,
                persist=True,
                device=self.device,
                verbose=False,
                conf=config.MIN_CONFIDENCE
            )

            # Flag to track presence of active tracks
            has_active_tracks = False
            active_tracks = set()

            if not results or len(results) == 0:
                # No detections - reset all sequences
                removed_count = self.sequence_tracker.reset_all_sequences()
                if removed_count > 0 and config.LOG_DETAILED_PREDICTIONS:
                    logger.debug(
                        f"[TRANSITION] No detections - reset all sequences, removed: {removed_count}")
            else:
                if results[0].boxes is None or results[0].boxes.id is None:
                    # Detections but no tracks - reset all sequences
                    removed_count = self.sequence_tracker.reset_all_sequences()
                    if removed_count > 0 and config.LOG_DETAILED_PREDICTIONS:
                        logger.debug(
                            f"[TRANSITION] No track IDs - reset all sequences, removed: {removed_count}")
                else:
                    has_active_tracks = True
                    annotated_frame = results[0].plot(boxes=False)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    keypoints = results[0].keypoints.xyn.cpu(
                    ).numpy() if results[0].keypoints else []

                    current_time = time.time()
                    active_tracks = set()

                    for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):

                        if conf < config.DETECTION_CONFIDENCE:
                            continue

                        self.detection_stats['detections'] += 1
                        x1, y1, x2, y2 = box.astype(int)
                        active_tracks.add(track_id)

                        # Expand bounding box
                        x1 = max(0, x1 - 30)
                        y1 = max(0, y1 - 10)
                        x2 = min(annotated_frame.shape[1], x2 + 30)
                        y2 = min(annotated_frame.shape[0], y2 + 10)

                        # Save detection information
                        detection_info = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'frame_width': config.DISPLAY_WIDTH,
                            'frame_height': config.DISPLAY_HEIGHT,
                            'timestamp': datetime.now().isoformat()
                        }

                        # Update recent detections list for this track_id
                        if track_id in self.last_detections:
                            self.last_detections[track_id].append(
                                detection_info)
                            # Limit list size
                            if len(self.last_detections[track_id]) > self.max_detections_per_track:
                                self.last_detections[track_id].pop(0)
                        else:
                            self.last_detections[track_id] = [detection_info]

                        if i < len(keypoints) and len(keypoints[i]) > 0:
                            # Prepare data for XGBoost classification
                            data = {}
                            for j, point in enumerate(keypoints[i]):
                                if not np.isnan(point[0]) and not np.isnan(point[1]):
                                    data[f'x{j}'] = point[0]
                                    data[f'y{j}'] = point[1]

                            if len(data) > 0:
                                df = pd.DataFrame(data, index=[0])
                                dmatrix = xgb.DMatrix(df)

                                # XGBoost prediction
                                prediction = self.model_xgb.predict(dmatrix)[0]
                                binary_prediction = int(
                                    prediction > config.SUSPICIOUS_CONFIDENCE_THRESHOLD)

                                # Log detailed predictions
                                # if config.LOG_DETAILED_PREDICTIONS and track_id == 1:
                                #     logger.debug(f"[MODEL] Track:{track_id}, "
                                #                  f"Raw:{prediction:.3f}, Binary:{binary_prediction}, "
                                #                  f"Result:{'NORMAL' if binary_prediction == 1 else 'SUSPICIOUS'}")

                                # Determine current state
                                is_suspicious = (binary_prediction == 0)
                                current_confidence = prediction

                                # Update sequence tracker
                                sequence_status = self.sequence_tracker.update(
                                    track_id, is_suspicious, current_confidence, current_time
                                )

                                # Checking sequence validation
                                is_newly_validated = (sequence_status['is_valid'] and
                                                      track_id not in self.validated_tracks)

                                # Processing validated behavior
                                if is_newly_validated:
                                    self._handle_validated_suspicious(track_id, sequence_status,
                                                                      prediction, frame, current_time)

                                # Determine visualization properties
                                label, color = self._determine_display_properties(
                                    track_id, is_suspicious, sequence_status, prediction
                                )

                                # Draw bounding box and text
                                cv2.rectangle(annotated_frame,
                                              (x1, y1), (x2, y2), color, 2)

                                # Different text based on environment
                                if config.MONITORING_ENV == 'test':
                                    sequence = self.sequence_tracker.sequences.get(
                                        track_id)
                                    if sequence:
                                        progress = f"{sequence.consecutive_count}/{config.SUSPICIOUS_CONSECUTIVE_THRESHOLD}"
                                        duration = f"{sequence_status['duration']:.1f}s" if sequence_status['duration'] > 0 else "0s"
                                        status = "VALID" if sequence_status[
                                            'is_valid'] else f"VAL:{progress}({duration})"
                                    else:
                                        status = "INIT"

                                    text = f"{label} ID:{track_id} {prediction:.2f} {status}"
                                else:
                                    text = f"{label} ID:{track_id}"

                                cvzone.putTextRect(
                                    annotated_frame, text, (x1, max(y1-10, 10)), 1, 1, colorR=color)

            # Check tracks that are no longer active
            current_time = time.time()
            tracks_to_remove = []

            for track_id in list(self.first_suspicious_time.keys()):
                if track_id not in active_tracks:
                    # If track not detected in current frame
                    if current_time - self.first_suspicious_time[track_id] > 3.0:
                        # Check if video should be saved on disappearance
                        sequence = self.sequence_tracker.sequences.get(
                            track_id)
                        if (sequence and sequence.is_valid() and track_id not in self.video_saved_for_track):

                            logger.info(
                                f"[DISAPPEARED] Track disappeared while VALIDATED Suspicious - Track:{track_id}")
                            self.save_evidence(
                                track_id, "Suspicious", 0.0, is_disappearance=True)

                        tracks_to_remove.append(track_id)

            # Remove processed tracks
            for track_id in tracks_to_remove:
                if track_id in self.first_suspicious_time:
                    del self.first_suspicious_time[track_id]
                if track_id in self.first_suspicious_frame:
                    del self.first_suspicious_frame[track_id]
                if track_id in self.image_saved_for_track:
                    self.image_saved_for_track.remove(track_id)
                if track_id in self.video_saved_for_track:
                    self.video_saved_for_track.remove(track_id)
                if track_id in self.current_state_was_suspicious:
                    del self.current_state_was_suspicious[track_id]
                if track_id in self.last_state_was_suspicious:
                    del self.last_state_was_suspicious[track_id]
                if track_id in self.validated_tracks:
                    self.validated_tracks.remove(track_id)
                if track_id in self.normal_start_time:
                    del self.normal_start_time[track_id]
                if track_id in self.normal_duration:
                    del self.normal_duration[track_id]

            # Clean up inactive sequences
            if has_active_tracks:
                removed_count = self.sequence_tracker.cleanup_inactive(
                    active_tracks)
                if removed_count > 0 and config.LOG_DETAILED_PREDICTIONS:
                    logger.debug(
                        f"[CLEAN] Cleaned up {removed_count} inactive sequences")
            else:
                # If no active tracks, reset all sequences
                removed_count = self.sequence_tracker.reset_all_sequences()
                if removed_count > 0 and config.LOG_DETAILED_PREDICTIONS:
                    logger.debug(
                        f"[TRANSITION] No active tracks - reset all {removed_count} sequences")

            # Periodic statistics logging
            # if self.frame_count % 100 == 0:
            #     self.log_validation_statistics()

            # Periodic sequence tracker monitoring
            # if self.frame_count % 100 == 0:
            #     self.monitor_sequence_tracker()

            return annotated_frame if has_active_tracks else frame_resized

        except Exception as e:
            logger.error(
                f"Error processing frame {self.frame_count}: {str(e)}")
            return frame_resized

    def print_stats(self):
        """Print detection statistics at completion."""
        logger.info("=== Detection Statistics ===")
        logger.info(f"Environment: {config.MONITORING_ENV}")
        logger.info(f"Camera: {config.CAMERA_NAME}")
        logger.info(f"Video Source: {config.VIDEO_SOURCE}")
        logger.info(f"Model: {config.XGBOOST_MODEL_PATH}")
        logger.info(f"Total frames: {self.detection_stats['total_frames']}")
        logger.info(
            f"Processed frames: {self.detection_stats['processed_frames']}")
        logger.info(f"Detections: {self.detection_stats['detections']}")
        logger.info(
            f"Valid sequences: {self.validation_stats['valid_sequences']}")

        # if config.MONITORING_ENV == 'test':
        #     logger.info(
        #         f"Test alerts recorded: {len(self.alert_manager.test_alerts)}")

    def run(self, video_source):
        """
        Main video processing loop.

        Args:
            video_source: Video source (file path, camera index, or RTSP stream)

        Returns:
            bool: True if processing completed successfully, False otherwise
        """
        logger.info(f"Starting detection in {config.MONITORING_ENV} mode")
        logger.info(f"Camera: {config.CAMERA_NAME}")
        logger.info(f"Video source: {video_source}")
        logger.info(f"XGBoost model: {config.XGBOOST_MODEL_PATH}")
        logger.info(
            f"Configuration: SAVE_EVIDENCE={config.SAVE_EVIDENCE}, SHOW_DISPLAY={config.SHOW_DISPLAY}")
        logger.info(
            f"Normal state threshold: {config.NORMAL_STATE_THRESHOLD} consecutive frames")
        logger.info(f"Suspicious validation: {config.SUSPICIOUS_CONSECUTIVE_THRESHOLD} consecutive frames, "
                    f"{config.SUSPICIOUS_CONFIDENCE_THRESHOLD} confidence, "
                    f"{config.MIN_SUSPICIOUS_DURATION}s minimum duration")

        # Initialize video source
        if isinstance(video_source, str) and video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
        else:
            cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            logger.error(f"Cannot open video source: {video_source}")
            return False

        logger.info("Video capture initialized successfully")

        output_path = f'out_{config.CAMERA_NAME}.mp4'
        out = None

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    if config.MONITORING_ENV == 'test':
                        logger.info("Test video ended - restarting")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                        continue
                    else:
                        logger.warning(
                            "End of video stream or cannot read frame")
                        break

                processed_frame = self.process_frame(frame)

                if out is None:
                    height, width = processed_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
                    out = cv2.VideoWriter(output_path, fourcc,
                                          config.OUTPUT_FPS, (width, height))
                    logger.info(f"VideoWriter initialized: {output_path}")

                if config.SHOW_DISPLAY:
                    # Add environment info to frame
                    cv2.putText(processed_frame,
                                f"Mode: {config.MONITORING_ENV} - Camera: {config.CAMERA_NAME} - Press 'q' to quit",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Calculate current statistics for display
                    try:
                        valid_count = len(
                            self.sequence_tracker.get_valid_sequences())
                        active_count = self.sequence_tracker.get_active_suspicious_count()

                        stats_text = f"Frames: {self.frame_count} | Active: {active_count} | Valid: {valid_count}"
                        cv2.putText(processed_frame, stats_text,
                                    (10, processed_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        # Fallback in case of error
                        stats_text = f"Frames: {self.frame_count}"
                        cv2.putText(processed_frame, stats_text,
                                    (10, processed_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        logger.debug(f"Error displaying stats: {str(e)}")

                    cv2.imshow(config.WINDOW_NAME, processed_frame)

                    if out is not None:
                        out.write(processed_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Manual stop by user")
                        break
                    elif key == ord('s'):  # Save screenshot on demand
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        cv2.imwrite(
                            f"{config.DEBUG_PATH}/screenshot_{timestamp}.jpg", processed_frame)
                        logger.info("Screenshot saved")
                    elif key == ord('p'):  # Pause on demand
                        logger.info("Video paused, press any key to continue")
                        cv2.waitKey(0)

                # Periodic logging in production mode
                if config.MONITORING_ENV == 'production' and self.frame_count % 300 == 0:
                    logger.info(f"Processing frame {self.frame_count}")

        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
        finally:
            cap.release()
            if out is not None:
                out.release()
                logger.info(f"Video saved: {output_path}")
            if config.SHOW_DISPLAY:
                cv2.destroyAllWindows()

            self.print_stats()
            return True


def main():
    """
    Main entry point for the shoplifting detection system.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        detector = ShopliftingDetector()
        success = detector.run(config.VIDEO_SOURCE)

        if success:
            logger.info("Detection completed successfully")
        else:
            logger.error("Detection failed")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
