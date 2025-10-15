import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, Scrollbar
from PIL import Image, ImageTk
import threading
import torch
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device for PyTorch (GPU if available, else CPU)
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
except Exception as e:
    logger.error(f"Error detecting device: {e}")
    device = 'cpu'

# Load YOLO model
try:
    model = YOLO("yolov8s.pt").to(device)
    names = model.model.names
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    sys.exit(1)


class BlurApp:
    """
    A GUI application for blurring specific tracked objects in video streams.

    This application uses YOLOv8 for object detection and tracking, allowing users
    to select specific track IDs to blur in real-time video processing. It features
    a dual-view display showing both original and processed video streams.

    Attributes:
        root (tk.Tk): The main Tkinter window
        capture (cv2.VideoCapture): Video capture object
        video_writer (cv2.VideoWriter): Video writer object for output
        running (bool): Flag indicating if video processing is active
        paused (bool): Flag indicating if video processing is paused
        frame (np.ndarray): Current video frame
        annotated_frame (np.ndarray): Current annotated video frame
        original_frame (np.ndarray): Current original video frame
        blur_mode (bool): Flag indicating if blurring is enabled
        selected_ids (set): Set of track IDs selected for blurring
        track_ids_ui (set): Set of track IDs displayed in UI
        checkbuttons (dict): Dictionary mapping track IDs to checkbutton widgets
        check_vars (dict): Dictionary mapping track IDs to IntVar variables
    """

    def __init__(self, root):
        """
        Initialize the BlurApp application.

        Args:
            root (tk.Tk): The main Tkinter window
        """
        self.root = root
        self.root.title("Track ID Blur Tool with Dual View")
        self.root.geometry("1920x1080")
        self.root.resizable(False, False)

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        # Bind space key for pause/resume
        self.root.bind('<Key-space>', self.toggle_pause)
        # Ensure the root window can receive key events
        self.root.focus_set()

        # Video processing attributes
        self.capture = None
        self.video_writer = None
        self.running = False
        self.paused = False
        self.frame = None
        self.annotated_frame = None
        self.original_frame = None
        self.blur_mode = False

        # Video output settings
        self.out_w, self.out_h = 930, 525
        self.fps = 30

        # Track ID management
        self.selected_ids = set()
        self.track_ids_ui = set()
        self.checkbuttons = {}
        self.check_vars = {}
        self.recent_track_ids = []  # List to maintain order of recent track IDs

        self.setup_ui()
        logger.info("BlurApp initialized successfully")

    def setup_ui(self):
        """Set up the user interface with video displays and control buttons."""
        try:
            # Create video display frame
            video_frame = tk.Frame(self.root)
            video_frame.pack(pady=5)

            # Original video label
            self.video_label_original = tk.Label(video_frame, text="Original Video",
                                                 borderwidth=2, relief="solid")
            self.video_label_original.pack(side="left", padx=5)

            # Processed video label
            self.video_label = tk.Label(video_frame, text="Processed Video",
                                        borderwidth=2, relief="solid")
            self.video_label.pack(side="left", padx=5)

            # Track ID selection frame with scrollbar
            track_id_frame = tk.Frame(self.root)
            track_id_frame.pack(padx=10, fill="x", pady=5)

            # Label for track ID section
            tk.Label(track_id_frame, text="Select Track IDs to Show (Blur):",
                     font=("Arial", 10, "bold")).pack(anchor="w")

            canvas = tk.Canvas(track_id_frame, height=60)
            h_scroll = Scrollbar(
                track_id_frame, orient="horizontal", command=canvas.xview)
            canvas.configure(xscrollcommand=h_scroll.set)

            h_scroll.pack(side="bottom", fill="x")
            canvas.pack(side="top", fill="x")

            self.track_id_inner = tk.Frame(canvas)
            canvas.create_window(
                (0, 0), window=self.track_id_inner, anchor="nw")
            self.track_id_inner.bind("<Configure>",
                                     lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

            # Control buttons frame
            btn_frame = tk.Frame(self.root)
            btn_frame.pack(pady=10)

            # Create control buttons
            buttons = [
                ("Start", self.start_video),
                ("Pause (Space)", self.pause_video),
                ("Resume (Space)", self.resume_video),
                ("Start Blurring", self.enable_blur),
                ("Stop Blurring", self.disable_blur),
                ("Quit", self.quit_app)
            ]

            for text, command in buttons:
                tk.Button(btn_frame, text=text, command=command,
                          width=18, height=1).pack(side="left", padx=5)

            # Status label
            self.status_label = tk.Label(self.root, text="Ready to start video",
                                         fg="blue", font=("Arial", 10))
            self.status_label.pack(pady=5)

            # Instruction label for space key
            space_label = tk.Label(self.root,
                                   text="Press SPACE to pause/resume video playback",
                                   fg="gray", font=("Arial", 9))
            space_label.pack(pady=2)

            logger.info("UI setup completed successfully")

        except Exception as e:
            logger.error(f"Error setting up UI: {e}")
            messagebox.showerror(
                "UI Error", f"Failed to set up user interface: {e}")

    def toggle_pause(self, event=None):
        """
        Toggle pause/resume state with space key.

        Args:
            event: Key event (optional)
        """
        if not self.running:
            return

        if self.paused:
            self.resume_video()
        else:
            self.pause_video()

    def update_track_id_checkboxes(self, track_ids):
        """
        Update the track ID checkboxes in the UI with new track IDs.

        Args:
            track_ids (list): List of track IDs to add to the UI
        """
        try:
            for track_id in track_ids:
                if track_id not in self.track_ids_ui:
                    self.track_ids_ui.add(track_id)
                    var = tk.IntVar()
                    cb = tk.Checkbutton(self.track_id_inner, text=f"ID {track_id}", variable=var,
                                        command=self.update_selected_ids)
                    cb.pack(side="left", padx=5)
                    self.checkbuttons[track_id] = cb
                    self.check_vars[track_id] = var

            logger.debug(
                f"Updated track ID checkboxes. Current IDs: {list(self.track_ids_ui)}")
        except Exception as e:
            logger.error(f"Error updating track ID checkboxes: {e}")

    def update_selected_ids(self):
        """Update the set of selected track IDs based on checkbox states."""
        try:
            self.selected_ids.clear()
            for track_id, var in self.check_vars.items():
                if var.get() == 1:
                    self.selected_ids.add(track_id)
            logger.debug(f"Selected IDs updated: {self.selected_ids}")
        except Exception as e:
            logger.error(f"Error updating selected IDs: {e}")

    def enable_blur(self):
        """Enable blurring mode for selected track IDs."""
        try:
            self.blur_mode = True
            messagebox.showinfo("Blur", "Blurring started for selected IDs.")
            logger.info("Blurring enabled")
        except Exception as e:
            logger.error(f"Error enabling blur: {e}")
            messagebox.showerror("Error", f"Failed to enable blur: {e}")

    def disable_blur(self):
        """Disable blurring mode."""
        try:
            self.blur_mode = False
            messagebox.showinfo("Blur", "Blurring stopped.")
            logger.info("Blurring disabled")
        except Exception as e:
            logger.error(f"Error disabling blur: {e}")
            messagebox.showerror("Error", f"Failed to disable blur: {e}")

    def pause_video(self):
        """Pause video processing."""
        if self.running and not self.paused:
            self.paused = True
            self.status_label.config(
                text="Video Paused (Press SPACE to resume)", fg="orange")
            logger.info("Video paused")

    def resume_video(self):
        """Resume video processing."""
        if self.running and self.paused:
            self.paused = False
            self.status_label.config(
                text="Video Playing (Press SPACE to pause)", fg="blue")
            logger.info("Video resumed")

    def quit_app(self):
        """Safely quit the application, releasing all resources."""
        try:
            self.running = False
            if self.capture:
                self.capture.release()
                logger.info("Video capture released")
            if self.video_writer:
                self.video_writer.release()
                logger.info("Video writer released")
            cv2.destroyAllWindows()
            self.root.quit()
            logger.info("Application terminated successfully")
        except Exception as e:
            logger.error(f"Error quitting application: {e}")
            # Force quit if there's an error
            self.root.quit()

    def auto_close_app(self, message="Processing completed"):
        """
        Automatically close the application with a message.

        Args:
            message (str): Message to display before closing
        """
        try:
            self.status_label.config(
                text=f"{message}. Closing...", fg="purple")
            logger.info(f"{message}. Initiating auto-close.")

            # Show message and close after a short delay
            self.root.after(2000, self.quit_app)  # Close after 2 seconds

        except Exception as e:
            logger.error(f"Error in auto_close_app: {e}")
            self.quit_app()

    def start_video(self):
        """Start video processing by initializing video capture and starting processing thread."""
        try:
            if self.running:
                messagebox.showinfo("Info", "Video is already running")
                return

            self.running = True
            self.paused = False
            self.selected_ids.clear()
            self.track_ids_ui.clear()
            self.recent_track_ids = []  # Reset recent track IDs

            # Clear existing checkbuttons from UI
            for widget in self.track_id_inner.winfo_children():
                widget.destroy()
            self.checkbuttons.clear()
            self.check_vars.clear()

            # Initialize video capture
            video_path = "../test_videos/test_cam1.mp4"  # Path to the test video file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            self.capture = cv2.VideoCapture(video_path)
            if not self.capture.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            # Get video properties
            self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            if self.fps <= 0:
                self.fps = 30
                logger.warning(
                    f"Invalid FPS detected, using default: {self.fps}")

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter("norm.mp4", fourcc,
                                                self.fps, (self.out_w, self.out_h))

            if not self.video_writer.isOpened():
                raise IOError("Cannot create video writer for output file.")

            self.status_label.config(
                text="Video Processing Started (Press SPACE to pause)", fg="green")

            # Start video processing in a separate thread
            threading.Thread(target=self.process_video, daemon=True).start()
            logger.info("Video processing started successfully")

        except FileNotFoundError as e:
            logger.error(f"Video file error: {e}")
            messagebox.showerror("File Error",
                                 f"Video file not found. Please ensure '{video_path}' exists.")
            self.running = False
            self.auto_close_app("Video file not found")
        except Exception as e:
            logger.error(f"Error starting video: {e}")
            messagebox.showerror("Error", f"Failed to start video: {e}")
            self.running = False
            self.auto_close_app("Failed to start video processing")

    def process_video(self):
        """
        Main video processing loop running in a separate thread.

        Handles frame reading, object detection, tracking, blurring, and display updates.
        Implements the logic:
        - If blur_mode is True: blur all except selected IDs
        """
        def update():
            """Process a single frame and schedule the next update."""
            if not self.running:
                return

            try:
                if not self.paused:
                    # Read frame from video capture
                    ret, frame = self.capture.read()
                    if not ret:
                        logger.info("End of video stream reached")
                        self.auto_close_app("Video processing completed")
                        return

                    # Resize frame to output dimensions
                    frame = cv2.resize(frame, (self.out_w, self.out_h))
                    self.original_frame = frame.copy()

                    # Perform object detection and tracking (only class 0 - persons)
                    results = model.track(
                        frame, persist=True, classes=[0], verbose=False)

                    if results and results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.int().cpu().tolist()
                        class_ids = results[0].boxes.cls.int().cpu().tolist()
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        self.update_track_id_checkboxes(track_ids)

                        # Process each detected object
                        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                            x1, y1, x2, y2 = box
                            # Expand bounding box for better blurring coverage
                            x1 = max(0, x1 - 30)
                            y1 = max(0, y1 - 10)
                            x2 = min(frame.shape[1], x2 + 30)
                            y2 = min(frame.shape[0], y2 + 10)

                            # Validate ROI dimensions
                            if x2 <= x1 or y2 <= y1:
                                continue

                            roi = frame[y1:y2, x1:x2]

                            # Skip if ROI is invalid
                            if roi.size == 0:
                                continue

                            if self.blur_mode and track_id in self.selected_ids:
                                # Apply blur to selected track ID
                                blur = cv2.blur(roi, (45, 45))
                                frame[y1:y2, x1:x2] = blur
                                cv2.putText(frame, f'ID:{track_id}', (x2 + 10, y1 + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                            else:
                                # Draw bounding box and ID for non-blurred objects
                                cv2.rectangle(frame, (x1, y1),
                                              (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f'ID:{track_id}', (x2 + 10, y1 + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    self.annotated_frame = frame.copy()
                    self.video_writer.write(frame)
                    self.display_frame(self.annotated_frame,
                                       self.original_frame)
                else:
                    # Display last frame when paused
                    if self.annotated_frame is not None:
                        self.display_frame(
                            self.annotated_frame, self.original_frame)

                # Schedule next frame processing
                delay = max(1, int(1000 / self.fps))
                self.root.after(delay, update)

            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                # Stop processing on critical error and close app
                self.running = False
                self.auto_close_app(
                    f"Error processing video frame: {str(e)[:100]}...")

        # Start the processing loop
        update()

    def display_frame(self, annotated, original):
        """
        Display frames in the GUI.

        Args:
            annotated (np.ndarray): Annotated frame to display
            original (np.ndarray): Original frame to display
        """
        def to_imgtk(cv_img):
            """Convert OpenCV image to PhotoImage for Tkinter display."""
            try:
                rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                return ImageTk.PhotoImage(pil_img)
            except Exception as e:
                logger.error(f"Error converting image for display: {e}")
                # Return a blank image on error
                blank_image = Image.new(
                    'RGB', (self.out_w, self.out_h), color='black')
                return ImageTk.PhotoImage(blank_image)

        try:
            self.imgtk_annotated = to_imgtk(annotated)
            self.imgtk_original = to_imgtk(original)

            self.video_label.config(image=self.imgtk_annotated)
            self.video_label.image = self.imgtk_annotated

            self.video_label_original.config(image=self.imgtk_original)
            self.video_label_original.image = self.imgtk_original
        except Exception as e:
            logger.error(f"Error displaying frames: {e}")


def main():
    """Main function to initialize and run the application."""
    try:
        root = tk.Tk()
        app = BlurApp(root)
        logger.info("Application started successfully")

        # Handle application close gracefully
        root.protocol("WM_DELETE_WINDOW", app.quit_app)

        root.mainloop()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        messagebox.showerror(
            "Fatal Error", f"Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
