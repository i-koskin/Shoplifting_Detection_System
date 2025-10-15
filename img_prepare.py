import os
import cv2
import shutil
import pandas as pd
import logging
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('images_prepare.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variable for storing values between functions
cropped_persons_normal = None


def process_normal_video(video_path, model_path, output_image_dir, output_crop_dir, output_csv_path, confidence_threshold=0.75):
    """
    Process normal video for human pose estimation using YOLO model.
    """
    global cropped_persons_normal

    # Create output directories if they don't exist
    try:
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_crop_dir, exist_ok=True)
        logger.info(
            f"Created output directories: {output_image_dir}, {output_crop_dir}")
    except Exception as e:
        logger.error(f"Error creating output directories: {e}")
        return None, None

    # Load YOLO model with error handling
    try:
        logger.info(f"Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None, None

    # Open video file with error handling
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
    except Exception as e:
        logger.error(f"Error opening video file: {e}")
        return None, None

    # Get video properties
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        logger.info(
            f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        logger.info(f"Target frames to process: {total_frames}")
    except Exception as e:
        logger.error(f"Error getting video properties: {e}")
        cap.release()
        return None, None

    # Initialize counters and data storage
    current_frame = 0
    current_crop_index = 0
    all_data = []

    try:
        with tqdm(total=total_frames, desc="Processing normal video frames") as pbar:
            while current_frame < total_frames and cap.isOpened():
                # Calculate position in milliseconds for frame sampling
                try:
                    position_ms = (current_frame *
                                   (duration / total_frames) * 1000)
                    cap.set(cv2.CAP_PROP_POS_MSEC, position_ms)
                except Exception as e:
                    logger.warning(
                        f"Error setting video position for frame {current_frame}: {e}")
                    current_frame += 1
                    continue

                # Read frame from video
                try:
                    success, frame = cap.read()
                    if not success:
                        logger.warning(f"Failed to read frame {current_frame}")
                        break
                except Exception as e:
                    logger.warning(f"Error reading frame {current_frame}: {e}")
                    continue

                # Save full frame image
                image_path = None
                try:
                    image_path = os.path.join(
                        output_image_dir, f'img_{current_frame:06d}.jpg')
                    cv2.imwrite(image_path, frame)
                except Exception as e:
                    logger.warning(
                        f"Error saving full frame {current_frame}: {e}")

                # Run YOLO pose estimation on current frame
                try:
                    results = model(frame, verbose=False)
                except Exception as e:
                    logger.warning(
                        f"YOLO inference error on frame {current_frame}: {e}")
                    # Delete the saved image if inference fails
                    if image_path and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            logger.debug(
                                f"Deleted image after inference error: {image_path}")
                        except Exception as delete_error:
                            logger.warning(
                                f"Error deleting image {image_path}: {delete_error}")
                    current_frame += 1
                    continue

                # Process detection results for each detected object
                for result in results:
                    try:
                        bounding_boxes = result.boxes.xyxy
                        confidence_scores = result.boxes.conf.tolist()
                        keypoints = result.keypoints.xyn.tolist()

                        # Process each detected person in the frame
                        for person_index, box in enumerate(bounding_boxes):
                            if (person_index < len(confidence_scores) and confidence_scores[person_index] > confidence_threshold):
                                try:
                                    # Extract bounding box coordinates
                                    x1, y1, x2, y2 = map(int, box.tolist())

                                    if x2 <= x1 or y2 <= y1:
                                        logger.warning(
                                            f"Invalid bounding box in frame {current_frame}")
                                        continue

                                    # Crop person from frame
                                    cropped_person = frame[y1:y2, x1:x2]

                                    # Skip if crop is too small
                                    if cropped_person.size == 0:
                                        continue

                                    # Save cropped person image
                                    crop_filename = f'person_crop_{current_crop_index:06d}.jpg'
                                    crop_path = os.path.join(
                                        output_crop_dir, crop_filename)
                                    cv2.imwrite(crop_path, cropped_person)

                                    # Prepare keypoint data
                                    person_data = {
                                        'image_name': crop_filename
                                    }

                                    # Add keypoints to data
                                    if (person_index < len(keypoints) and
                                            len(keypoints[person_index]) > 0):
                                        for kp_idx, keypoint in enumerate(keypoints[person_index]):
                                            if kp_idx < len(keypoints[person_index]):
                                                person_data[f'x{kp_idx}'] = keypoint[0]
                                                person_data[f'y{kp_idx}'] = keypoint[1]

                                    all_data.append(person_data)
                                    current_crop_index += 1

                                except Exception as e:
                                    logger.warning(
                                        f"Error processing person {person_index} in frame {current_frame}: {e}")
                                    continue

                    except Exception as e:
                        logger.warning(
                            f"Error processing results for frame {current_frame}: {e}")
                        continue

                # Delete the processed image file after successful processing
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        logger.debug(f"Deleted processed image: {image_path}")
                    except Exception as e:
                        logger.warning(
                            f"Error deleting image {image_path}: {e}")

                current_frame += 1
                pbar.update(1)
                pbar.set_postfix({
                    'frames': current_frame,
                    'persons_detected': current_crop_index
                })

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Clean up any remaining images in output_image_dir
    try:
        if os.path.exists(output_image_dir):
            remaining_files = [f for f in os.listdir(
                output_image_dir) if f.endswith('.jpg')]
            if remaining_files:
                logger.info(
                    f"Cleaning up {len(remaining_files)} remaining images in {output_image_dir}")
                for file in remaining_files:
                    file_path = os.path.join(output_image_dir, file)
                    os.remove(file_path)
                logger.info("Cleanup completed")
    except Exception as e:
        logger.warning(f"Error during cleanup of {output_image_dir}: {e}")

    # Save keypoint data to CSV file
    if all_data:
        try:
            df = pd.DataFrame(all_data)
            if not os.path.isfile(output_csv_path):
                df.to_csv(output_csv_path, index=False)
                logger.info(f"Created new CSV file: {output_csv_path}")
            else:
                df.to_csv(output_csv_path, mode='a', header=False, index=False)
                logger.info(
                    f"Appended data to existing CSV file: {output_csv_path}")
        except Exception as e:
            logger.error(f"Error saving CSV data: {e}")
            return None, None

    # Prepare and return processing statistics
    stats = {
        'processed_frames': current_frame,
        'cropped_persons': current_crop_index,
        'total_frames': total_frames
    }

    # Save the value to a global variable
    cropped_persons_normal = current_crop_index

    logger.info(
        f"Processing completed: {current_frame} frames processed, {current_crop_index} persons detected")
    return cropped_persons_normal, stats


def process_suspicious_video(video_path, model_path, output_image_dir, output_crop_dir, output_csv_path,
                             start_crop_index, confidence_threshold=0.75):
    """
    Process suspicious video for human pose estimation, continuing from previous crop index.
    """
    # Create output directories if they don't exist
    try:
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_crop_dir, exist_ok=True)
        logger.info(
            f"Output directories verified: {output_image_dir}, {output_crop_dir}")
    except Exception as e:
        logger.error(f"Error creating output directories: {e}")
        return None

    # Load YOLO model with error handling
    try:
        logger.info(f"Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None

    # Open video file with error handling
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        logger.info(f"Video file opened successfully: {video_path}")
    except Exception as e:
        logger.error(f"Error opening video file: {e}")
        return None

    # Get video properties
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        logger.info(
            f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        logger.info(f"Target frames to process: {total_frames}")
    except Exception as e:
        logger.error(f"Error getting video properties: {e}")
        cap.release()
        return None

    # Initialize counters and data storage
    current_frame = 0
    current_crop_index = start_crop_index
    all_data = []

    try:
        with tqdm(total=total_frames, desc="Processing suspicious video frames") as pbar:
            while current_frame < total_frames and cap.isOpened():
                # Calculate position in milliseconds for uniform sampling
                try:
                    position_ms = (current_frame *
                                   (duration / total_frames) * 1000)
                    cap.set(cv2.CAP_PROP_POS_MSEC, position_ms)
                except Exception as e:
                    logger.warning(
                        f"Error setting video position for frame {current_frame}: {e}")
                    current_frame += 1
                    continue

                # Read frame from video
                try:
                    success, frame = cap.read()
                    if not success:
                        logger.warning(
                            f"Failed to read frame {current_frame}, ending processing")
                        break
                except Exception as e:
                    logger.warning(f"Error reading frame {current_frame}: {e}")
                    current_frame += 1
                    continue

                # Save full frame image
                image_path = None
                try:
                    image_path = os.path.join(
                        output_image_dir, f'img_{current_frame:06d}.jpg')
                    cv2.imwrite(image_path, frame)
                except Exception as e:
                    logger.warning(
                        f"Error saving full frame {current_frame}: {e}")

                # Run YOLO pose estimation on current frame
                try:
                    results = model(frame, verbose=False)
                except Exception as e:
                    logger.warning(
                        f"YOLO inference error on frame {current_frame}: {e}")
                    # Delete the saved image if inference fails
                    if image_path and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            logger.debug(
                                f"Deleted image after inference error: {image_path}")
                        except Exception as delete_error:
                            logger.warning(
                                f"Error deleting image {image_path}: {delete_error}")
                    current_frame += 1
                    continue

                # Process detection results for each detected object
                for result in results:
                    try:
                        bounding_boxes = result.boxes.xyxy
                        confidence_scores = result.boxes.conf.tolist()
                        keypoints = result.keypoints.xyn.tolist()

                        # Process each detected person in the frame
                        for person_index, box in enumerate(bounding_boxes):
                            if (person_index < len(confidence_scores) and
                                    confidence_scores[person_index] > confidence_threshold):
                                try:
                                    x1, y1, x2, y2 = map(int, box.tolist())

                                    if x2 <= x1 or y2 <= y1:
                                        logger.warning(
                                            f"Invalid bounding box in frame {current_frame}")
                                        continue

                                    # Crop person from frame
                                    cropped_person = frame[y1:y2, x1:x2]

                                    # Skip if crop is too small
                                    if cropped_person.size == 0:
                                        continue

                                    crop_filename = f'person_crop_{current_crop_index:06d}.jpg'
                                    crop_path = os.path.join(
                                        output_crop_dir, crop_filename)
                                    cv2.imwrite(crop_path, cropped_person)

                                    person_data = {
                                        'image_name': crop_filename
                                    }

                                    if (person_index < len(keypoints) and
                                            len(keypoints[person_index]) > 0):
                                        for kp_idx, keypoint in enumerate(keypoints[person_index]):
                                            if kp_idx < len(keypoints[person_index]):
                                                person_data[f'x{kp_idx}'] = keypoint[0]
                                                person_data[f'y{kp_idx}'] = keypoint[1]

                                    all_data.append(person_data)
                                    current_crop_index += 1

                                except Exception as e:
                                    logger.warning(
                                        f"Error processing person {person_index} in frame {current_frame}: {e}")
                                    continue

                    except Exception as e:
                        logger.warning(
                            f"Error processing results for frame {current_frame}: {e}")
                        continue

                # Delete the processed image file after successful processing
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        logger.debug(f"Deleted processed image: {image_path}")
                    except Exception as e:
                        logger.warning(
                            f"Error deleting image {image_path}: {e}")

                current_frame += 1
                pbar.update(1)
                pbar.set_postfix({
                    'frames': current_frame,
                    'persons_detected': current_crop_index - start_crop_index,
                    'current_crop': current_crop_index
                })

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Video resources released")

    # Clean up any remaining images in output_image_dir
    try:
        if os.path.exists(output_image_dir):
            remaining_files = [f for f in os.listdir(
                output_image_dir) if f.endswith('.jpg')]
            if remaining_files:
                logger.info(
                    f"Cleaning up {len(remaining_files)} remaining images in {output_image_dir}")
                for file in remaining_files:
                    file_path = os.path.join(output_image_dir, file)
                    os.remove(file_path)
                logger.info("Cleanup completed")
    except Exception as e:
        logger.warning(f"Error during cleanup of {output_image_dir}: {e}")

    # Save keypoint data to CSV file
    if all_data:
        try:
            df = pd.DataFrame(all_data)
            if not os.path.isfile(output_csv_path):
                df.to_csv(output_csv_path, index=False)
                logger.info(f"Created new CSV file: {output_csv_path}")
            else:
                df.to_csv(output_csv_path, mode='a', header=False, index=False)
                logger.info(
                    f"Appended {len(df)} records to existing CSV: {output_csv_path}")
        except Exception as e:
            logger.error(f"Error saving CSV data: {e}")
            return None
    else:
        logger.warning("No person detection data to save")

    # Prepare and return processing statistics
    stats = {
        'processed_frames': current_frame,
        'total_persons_detected': current_crop_index - start_crop_index,
        'final_crop_index': current_crop_index,
        'total_frames': total_frames
    }

    logger.info(f"Processing completed: {current_frame}/{total_frames} frames processed, "
                f"{stats['total_persons_detected']} persons detected")
    return stats


class FileOrganizer:
    """
    A class to organize image files into categories based on their numbering.
    """

    def __init__(self, source_directory, normal_folder, suspicious_folder):
        self.source_directory = source_directory
        self.normal_folder = normal_folder
        self.suspicious_folder = suspicious_folder
        self.source_path = Path(source_directory)
        self.normal_path = Path(normal_folder)
        self.suspicious_path = Path(suspicious_folder)

        logger.info(f"FileOrganizer initialized with:")
        logger.info(f"  Source: {source_directory}")
        logger.info(f"  Normal: {normal_folder}")
        logger.info(f"  Suspicious: {suspicious_folder}")

    def validate_directories(self):
        """
        Validate that all required directories exist and are accessible.
        """
        try:
            if not self.source_path.exists():
                logger.error(
                    f"Source directory does not exist: {self.source_directory}")
                return False

            if not self.source_path.is_dir():
                logger.error(
                    f"Source path is not a directory: {self.source_directory}")
                return False

            self.normal_path.mkdir(parents=True, exist_ok=True)
            self.suspicious_path.mkdir(parents=True, exist_ok=True)

            if not os.access(self.normal_path, os.W_OK):
                logger.error(
                    f"No write permission for normal folder: {self.normal_folder}")
                return False

            if not os.access(self.suspicious_path, os.W_OK):
                logger.error(
                    f"No write permission for suspicious folder: {self.suspicious_folder}")
                return False

            logger.info("All directories validated successfully")
            return True

        except Exception as e:
            logger.error(f"Error validating directories: {e}")
            return False

    def extract_number_from_filename(self, file_name):
        """
        Extract numeric value from filename following the pattern 'person_crop_<number>.jpg'
        """
        try:
            if not file_name.startswith('person_crop_'):
                logger.debug(
                    f"Filename '{file_name}' does not match expected pattern")
                return None

            parts = file_name.split('_')
            if len(parts) < 3:
                logger.warning(f"Filename '{file_name}' has unexpected format")
                return None

            number_part = parts[2].split('.')[0]
            number = int(number_part)
            logger.debug(
                f"Extracted number {number} from filename '{file_name}'")
            return number

        except ValueError as e:
            logger.warning(
                f"Could not extract number from filename '{file_name}': {e}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error processing filename '{file_name}': {e}")
            return None

    def categorize_file(self, file_name, number):
        """
        Determine the target category for a file based on its number.
        Uses the global cropped_persons_normal value.
        """
        global cropped_persons_normal

        try:
            if cropped_persons_normal is None:
                logger.error(
                    "cropped_persons_normal is not set. Cannot categorize files.")
                return 'suspicious'

            if 0 <= number <= cropped_persons_normal:
                category = 'normal'
            else:
                category = 'suspicious'

            # logger.debug(f"File '{file_name}' with number {number} categorized as '{category}'")
            return category

        except Exception as e:
            logger.error(f"Error categorizing file '{file_name}': {e}")
            return 'suspicious'

    def move_file(self, source_file_path, target_folder, file_name):
        """
        Safely move a file from source to target directory.
        """
        try:
            target_file_path = target_folder / file_name

            if target_file_path.exists():
                logger.warning(
                    f"Target file already exists: {target_file_path}")
                base_name = target_file_path.stem
                extension = target_file_path.suffix
                counter = 1
                while target_file_path.exists():
                    new_name = f"{base_name}_{counter}{extension}"
                    target_file_path = target_folder / new_name
                    counter += 1
                logger.info(
                    f"Using alternative filename: {target_file_path.name}")

            shutil.move(str(source_file_path), str(target_file_path))
            # logger.info(f"Successfully moved '{file_name}' to {target_folder.name} folder")
            return True

        except PermissionError as e:
            logger.error(f"Permission denied when moving '{file_name}': {e}")
            return False
        except FileNotFoundError as e:
            logger.error(f"Source file not found '{file_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error moving '{file_name}': {e}")
            return False

    def organize_files(self):
        """
        Main method to organize all files in the source directory.
        """
        statistics = {
            'total_files': 0,
            'processed_files': 0,
            'normal_files': 0,
            'suspicious_files': 0,
            'skipped_files': 0,
            'errors': 0
        }

        try:
            if not self.validate_directories():
                logger.error("Directory validation failed. Cannot proceed.")
                statistics['errors'] += 1
                return statistics

            all_files = [f for f in self.source_path.iterdir() if f.is_file()]
            statistics['total_files'] = len(all_files)

            if statistics['total_files'] == 0:
                logger.warning("No files found in source directory")
                return statistics

            logger.info(
                f"Starting to process {statistics['total_files']} files")

            matching_files = [
                f for f in all_files if f.name.startswith('person_crop_')]
            non_matching_files = [
                f for f in all_files if not f.name.startswith('person_crop_')]

            statistics['skipped_files'] = len(non_matching_files)

            if non_matching_files:
                logger.info(
                    f"Skipping {len(non_matching_files)} files that don't match pattern")

            # We use the global value cropped_persons_normal
            global cropped_persons_normal
            if cropped_persons_normal is not None:
                logger.info(
                    f"Using cropped_persons_normal value: {cropped_persons_normal}")
            else:
                logger.warning("cropped_persons_normal is not set!")

            with tqdm(total=len(matching_files), desc="Organizing files", unit="file") as pbar:
                for file_path in matching_files:
                    file_name = file_path.name

                    try:
                        number = self.extract_number_from_filename(file_name)

                        if number is None:
                            statistics['skipped_files'] += 1
                            logger.warning(
                                f"Skipping file '{file_name}' - could not extract number")
                            pbar.update(1)
                            continue

                        category = self.categorize_file(file_name, number)

                        if category == 'normal':
                            target_folder = self.normal_path
                            statistics['normal_files'] += 1
                        else:
                            target_folder = self.suspicious_path
                            statistics['suspicious_files'] += 1

                        if self.move_file(file_path, target_folder, file_name):
                            statistics['processed_files'] += 1
                        else:
                            statistics['errors'] += 1

                    except Exception as e:
                        logger.error(
                            f"Error processing file '{file_name}': {e}")
                        statistics['errors'] += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        'Normal': statistics['normal_files'],
                        'Suspicious': statistics['suspicious_files'],
                        'Errors': statistics['errors']
                    })

            logger.info("File organization completed")
            logger.info(
                f"Total files processed: {statistics['processed_files']}")
            logger.info(f"Files moved to Normal: {statistics['normal_files']}")
            logger.info(
                f"Files moved to Suspicious: {statistics['suspicious_files']}")
            logger.info(f"Files skipped: {statistics['skipped_files']}")
            logger.info(f"Errors encountered: {statistics['errors']}")

            return statistics

        except Exception as e:
            logger.error(f"Unexpected error during file organization: {e}")
            statistics['errors'] += 1
            return statistics


def main():
    """
    Main function that processes videos and organizes files.
    """
    global cropped_persons_normal

    try:
        # Configuration for video processing
        NORMAL_CONFIG = {
            'video_path': './norm.mp4',
            'model_path': "yolo11s-pose.pt",
            'output_image_dir': './images',
            'output_crop_dir': './images_crop',
            'output_csv_path': './nkeypoints.csv',
            'confidence_threshold': 0.75
        }

        print("=== Normal Video Processing ===")
        print(f"Video: {NORMAL_CONFIG['video_path']}")
        print("-" * 50)

        # Process normal video
        cropped_persons_normal, normal_results = process_normal_video(
            **NORMAL_CONFIG)

        if not normal_results:
            print("Error processing normal video. Exiting.")
            return False

        # Configuration for suspicious video processing
        SUSPICIOUS_CONFIG = {
            'video_path': './susp.mp4',
            'model_path': "yolo11s-pose.pt",
            'output_image_dir': './images',
            'output_crop_dir': './images_crop',
            'output_csv_path': './nkeypoints.csv',
            'start_crop_index': cropped_persons_normal + 1,
            'confidence_threshold': 0.75
        }

        print("\n=== Suspicious Video Processing ===")
        print(f"Video: {SUSPICIOUS_CONFIG['video_path']}")
        print(
            f"Starting crop index: {SUSPICIOUS_CONFIG['start_crop_index']} (normal persons + 1)")
        print("-" * 50)

        # Process suspicious video
        suspicious_results = process_suspicious_video(**SUSPICIOUS_CONFIG)

        if not suspicious_results:
            print("Error processing suspicious video. Exiting.")
            return False

        # Display combined summary
        print("\n=== Processing Summary ===")
        print("Normal Video:")
        print(
            f"  Frames processed: {normal_results['processed_frames']}/{normal_results['total_frames']}")
        print(f"  Persons detected: {normal_results['cropped_persons']}")

        if suspicious_results:
            print("Suspicious Video:")
            print(
                f"  Frames processed: {suspicious_results['processed_frames']}/{suspicious_results['total_frames']}")
            print(
                f"  Persons detected: {suspicious_results['total_persons_detected']}")
            print(
                f"  Final crop index: {suspicious_results['final_crop_index']}")

        print(f"\nOutput CSV: {NORMAL_CONFIG['output_csv_path']}")
        print("Video processing completed successfully!")

        # Now organize the files
        print("\n" + "="*50)
        print("FILE ORGANIZATION")
        print("="*50)

        # Configuration for file organization
        source_directory = './images_crop'
        normal_folder = './dataset_path/normal'
        suspicious_folder = './dataset_path/suspicious'

        logger.info("Starting file organization process")

        # Create organizer instance
        organizer = FileOrganizer(
            source_directory, normal_folder, suspicious_folder)

        # Execute organization
        statistics = organizer.organize_files()

        # Print summary
        print("\n" + "="*50)
        print("FILE ORGANIZATION SUMMARY")
        print("="*50)
        print(f"Total files found: {statistics['total_files']}")
        print(f"Successfully processed: {statistics['processed_files']}")
        print(f"  - Normal category: {statistics['normal_files']}")
        print(f"  - Suspicious category: {statistics['suspicious_files']}")
        print(f"Skipped files: {statistics['skipped_files']}")
        print(f"Errors: {statistics['errors']}")

        if statistics['errors'] == 0:
            print("\nProcessing completed successfully!")
            return True
        else:
            print("\nProcessing completed with errors. Check log file for details.")
            return False

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Fatal error in main_combined: {e}")
        print(f"\nFatal error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
