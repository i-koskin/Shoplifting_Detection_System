import os
import pandas as pd
import logging
import sys
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_labeling.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatasetLabeler:
    """
    A class for labeling datasets based on image file presence in directories.

    This class provides functionality to read CSV files containing image metadata,
    validate directory structures, and assign labels based on whether images
    are found in 'suspicious' or 'normal' directories.
    """

    def __init__(self):
        """Initialize the DatasetLabeler with default settings."""
        self.processed_count = 0
        self.suspicious_count = 0
        self.normal_count = 0
        self.unlabeled_count = 0

    def read_csv_file(self, file_path):
        """
        Read CSV file with comprehensive error handling and validation.

        Args:
            file_path (str): Path to the CSV file to read

        Returns:
            pandas.DataFrame: Loaded DataFrame

        Raises:
            FileNotFoundError: If CSV file does not exist
            pd.errors.EmptyDataError: If CSV file is empty
            Exception: For other reading errors
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            logger.info(f"Reading CSV file: {file_path}")

            # Show progress for large CSV files
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.info(
                    f"Large file detected: {file_size / (1024*1024):.2f} MB")

            df = pd.read_csv(file_path)

            if df.empty:
                logger.warning("CSV file is empty")
            else:
                logger.info(
                    f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")

            return df

        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty CSV file error: {e}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading CSV file: {e}")
            raise

    def validate_directory_path(self, path, path_name="directory"):
        """
        Validate that a directory path exists and is accessible.

        Args:
            path (str): Path to validate
            path_name (str): Descriptive name for the path (for error messages)

        Returns:
            bool: True if path is valid, False otherwise
        """
        try:
            if not os.path.exists(path):
                logger.error(f"{path_name} path does not exist: {path}")
                return False

            if not os.path.isdir(path):
                logger.error(f"{path_name} path is not a directory: {path}")
                return False

            # Test if directory is readable
            if not os.access(path, os.R_OK):
                logger.error(f"{path_name} path is not readable: {path}")
                return False

            # Count files in directory for progress estimation
            file_count = len([f for f in os.listdir(
                path) if os.path.isfile(os.path.join(path, f))])
            logger.info(
                f"Validated {path_name} path: {path} ({file_count} files)")
            return True

        except Exception as e:
            logger.error(f"Error validating {path_name} path {path}: {e}")
            return False

    def get_label(self, image_name, susp_path, normal_path):
        """
        Determine the label for an image based on its presence in directory structures.

        Args:
            image_name (str): Name of the image file
            susp_path (str): Path to suspicious images directory
            normal_path (str): Path to normal images directory

        Returns:
            str: 'Suspicious' if image found in suspicious directory,
                 'Normal' if found in normal directory,
                 None if not found in either directory
        """
        try:
            # Validate that image_name is a string and not empty
            if not isinstance(image_name, str) or not image_name.strip():
                logger.warning(f"Invalid image name: {image_name}")
                self.unlabeled_count += 1
                return None

            # Check if image exists in suspicious directory
            sus_image_path = os.path.join(susp_path, image_name)
            if os.path.exists(sus_image_path):
                logger.debug(f"Image {image_name} classified as Suspicious")
                self.suspicious_count += 1
                return 'Suspicious'

            # Check if image exists in normal directory
            normal_image_path = os.path.join(normal_path, image_name)
            if os.path.exists(normal_image_path):
                logger.debug(f"Image {image_name} classified as Normal")
                self.normal_count += 1
                return 'Normal'

            # Image not found in either directory
            logger.warning(
                f"Image {image_name} not found in either suspicious or normal directories")
            self.unlabeled_count += 1
            return None

        except Exception as e:
            logger.error(f"Error classifying image {image_name}: {e}")
            self.unlabeled_count += 1
            return None

    def count_images_in_directories(self, susp_path, normal_path):
        """
        Count images in suspicious and normal directories for verification.

        Args:
            susp_path (str): Path to suspicious images directory
            normal_path (str): Path to normal images directory

        Returns:
            tuple: (suspicious_count, normal_count, total_count)
        """
        try:
            # Use tqdm for directory scanning if directories are large
            susp_images = []
            normal_images = []

            if os.path.exists(susp_path):
                files = os.listdir(susp_path)
                for file in tqdm(files, desc="Scanning suspicious directory", unit="file"):
                    if os.path.isfile(os.path.join(susp_path, file)):
                        susp_images.append(file)

            if os.path.exists(normal_path):
                files = os.listdir(normal_path)
                for file in tqdm(files, desc="Scanning normal directory", unit="file"):
                    if os.path.isfile(os.path.join(normal_path, file)):
                        normal_images.append(file)

            susp_count = len(susp_images)
            normal_count = len(normal_images)
            total_count = susp_count + normal_count

            logger.info(
                f"Found {susp_count} suspicious images and {normal_count} normal images (total: {total_count})")
            return susp_count, normal_count, total_count

        except Exception as e:
            logger.error(f"Error counting images in directories: {e}")
            return 0, 0, 0

    def add_labels_to_dataframe(self, df, susp_path, normal_path):
        """
        Add label column to DataFrame based on image presence in directories.

        Args:
            df (pandas.DataFrame): Input DataFrame with image names
            susp_path (str): Path to suspicious images directory
            normal_path (str): Path to normal images directory

        Returns:
            pandas.DataFrame: DataFrame with added 'label' column

        Raises:
            ValueError: If DataFrame doesn't contain required columns
            Exception: For other processing errors
        """
        try:
            # Validate input DataFrame
            if df.empty:
                logger.warning("DataFrame is empty, no labels to add")
                return df

            if 'image_name' not in df.columns:
                raise ValueError("DataFrame must contain 'image_name' column")

            # Count images for verification
            susp_count, normal_count, total_count = self.count_images_in_directories(
                susp_path, normal_path)

            # Reset counters
            self.processed_count = 0
            self.suspicious_count = 0
            self.normal_count = 0
            self.unlabeled_count = 0

            # Apply labeling function with progress bar
            logger.info("Starting to label images...")
            start_time = time.time()

            # Create a custom apply function with progress bar
            def label_with_progress(image_name):
                result = self.get_label(image_name, susp_path, normal_path)
                self.processed_count += 1
                return result

            # Use tqdm to show progress for the labeling process
            tqdm.pandas(desc="Labeling images", unit="image")
            df['label'] = df['image_name'].progress_apply(
                lambda x: label_with_progress(x)
            )

            labeling_time = time.time() - start_time
            logger.info(f"Labeling completed in {labeling_time:.2f} seconds")

            # Log labeling statistics
            logger.info(
                f"Labeling results: {self.suspicious_count} suspicious, "
                f"{self.normal_count} normal, {self.unlabeled_count} unlabeled"
            )

            # Verify counts match
            if self.suspicious_count != susp_count:
                logger.warning(
                    f"Labeled suspicious count ({self.suspicious_count}) doesn't match directory count ({susp_count})")

            if self.normal_count != normal_count:
                logger.warning(
                    f"Labeled normal count ({self.normal_count}) doesn't match directory count ({normal_count})")

            # Calculate and log labeling rate
            total_labeled = self.suspicious_count + self.normal_count
            labeling_rate = (total_labeled / len(df)) * \
                100 if len(df) > 0 else 0
            logger.info(f"Labeling rate: {labeling_rate:.1f}%")

            return df

        except Exception as e:
            logger.error(f"Error adding labels to DataFrame: {e}")
            raise

    def save_dataframe(self, df, output_path):
        """
        Save DataFrame to CSV file with comprehensive error handling.

        Args:
            df (pandas.DataFrame): DataFrame to save
            output_path (str): Path for output CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if df.empty:
                logger.warning("Attempting to save empty DataFrame")

            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")

            # Show progress for large DataFrames
            if len(df) > 10000:
                logger.info(f"Saving large DataFrame with {len(df)} rows...")

            # Save with progress indication
            with tqdm(total=1, desc="Saving DataFrame", unit="file") as pbar:
                df.to_csv(output_path, index=False)
                pbar.update(1)

            logger.info(f"Successfully saved DataFrame to: {output_path}")

            # Verify the file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Output file size: {file_size / 1024:.2f} KB")
                return True
            else:
                logger.error("Output file was not created")
                return False

        except PermissionError as e:
            logger.error(
                f"Permission denied when saving to {output_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error saving DataFrame to {output_path}: {e}")
            return False

    def generate_summary_report(self, df, output_path):
        """
        Generate a detailed summary report of the labeling process.

        Args:
            df (pandas.DataFrame): Labeled DataFrame
            output_path (str): Path for the summary report

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            report_content = []
            report_content.append("DATASET LABELING SUMMARY REPORT")
            report_content.append("=" * 50)
            report_content.append(f"Generated at: {pd.Timestamp.now()}")
            report_content.append(f"Total images processed: {len(df)}")
            report_content.append(
                f"Suspicious images: {self.suspicious_count}")
            report_content.append(f"Normal images: {self.normal_count}")
            report_content.append(f"Unlabeled images: {self.unlabeled_count}")

            labeling_rate = (
                (self.suspicious_count + self.normal_count) / len(df)) * 100
            report_content.append(f"Labeling rate: {labeling_rate:.1f}%")

            report_content.append("\nLABEL DISTRIBUTION:")
            label_counts = df['label'].value_counts()
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                report_content.append(
                    f"  {label}: {count} ({percentage:.1f}%)")

            # Save report
            report_path = output_path.replace('.csv', '_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))

            logger.info(f"Summary report saved to: {report_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return False

    def main(self):
        """
        Execute the complete dataset labeling workflow.

        Workflow:
        1. Read input CSV file with keypoint data
        2. Validate directory paths for suspicious and normal images
        3. Add labels based on image presence in directories
        4. Save labeled dataset to new CSV file
        5. Generate summary report

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Starting dataset labeling process")
            start_time = time.time()

            # Configuration
            input_csv_path = './nkeypoints.csv'
            dataset_path = './dataset_path'
            susp_path = os.path.join(dataset_path, 'suspicious')
            normal_path = os.path.join(dataset_path, 'normal')
            output_csv_path = os.path.join(dataset_path, 'dataset.csv')

            logger.info(f"Input CSV: {input_csv_path}")
            logger.info(f"Dataset path: {dataset_path}")
            logger.info(f"Suspicious images path: {susp_path}")
            logger.info(f"Normal images path: {normal_path}")
            logger.info(f"Output CSV: {output_csv_path}")

            # Step 1: Read input CSV
            logger.info("\n" + "="*50)
            logger.info("STEP 1: READING INPUT CSV")
            logger.info("="*50)
            df = self.read_csv_file(input_csv_path)

            # Step 2: Validate directory paths
            logger.info("\n" + "="*50)
            logger.info("STEP 2: VALIDATING DIRECTORY PATHS")
            logger.info("="*50)
            if not self.validate_directory_path(susp_path, "Suspicious images"):
                return False

            if not self.validate_directory_path(normal_path, "Normal images"):
                return False

            # Step 3: Add labels to DataFrame
            logger.info("\n" + "="*50)
            logger.info("STEP 3: LABELING IMAGES")
            logger.info("="*50)
            df_labeled = self.add_labels_to_dataframe(
                df, susp_path, normal_path)

            # Step 4: Save results
            logger.info("\n" + "="*50)
            logger.info("STEP 4: SAVING RESULTS")
            logger.info("="*50)
            if self.save_dataframe(df_labeled, output_csv_path):
                # Step 5: Generate summary report
                logger.info("\n" + "="*50)
                logger.info("STEP 5: GENERATING SUMMARY REPORT")
                logger.info("="*50)
                self.generate_summary_report(df_labeled, output_csv_path)

                total_time = time.time() - start_time
                logger.info(
                    f"\nDataset labeling process completed successfully in {total_time:.2f} seconds")

                # Final statistics
                total_images = len(df_labeled)
                labeled_images = self.suspicious_count + self.normal_count
                labeling_rate = (labeled_images / total_images) * \
                    100 if total_images > 0 else 0

                logger.info(f"FINAL STATISTICS:")
                logger.info(f"  Total images: {total_images}")
                logger.info(
                    f"  Labeled images: {labeled_images} ({labeling_rate:.1f}%)")
                logger.info(f"  Suspicious: {self.suspicious_count}")
                logger.info(f"  Normal: {self.normal_count}")
                logger.info(f"  Unlabeled: {self.unlabeled_count}")

                return True
            else:
                logger.error("Failed to save labeled dataset")
                return False

        except Exception as e:
            logger.error(f"Dataset labeling process failed: {e}")
            return False


def main():
    """
    Main entry point for the dataset labeling script.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        labeler = DatasetLabeler()
        success = labeler.main()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
