import cv2
import os
import csv
import random
import mediapipe as mp

# Folder with non-seizure videos
video_folder = './no_seizure_class'

# Output CSV path
output_csv_path = './non_seizure_pose_landmarks_all.csv'

# Randomly select 480 videos from the folder
all_videos = [f for f in os.listdir(
    video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
random.shuffle(all_videos)
selected_videos = all_videos[:480]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open CSV for writing
with open(output_csv_path, mode='w', newline='') as f:
    csv_writer = csv.writer(f)

    # Write header
    csv_writer.writerow([
        'video_name', 'frame', 'landmark_index',
        'x', 'y', 'z', 'visibility', 'label'
    ])

    # Process selected videos
    for video_file in selected_videos:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        print(f"Processing: {video_file}")
        frame_count = 0

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    csv_writer.writerow([
                        video_file,
                        frame_count,
                        idx,
                        lm.x,
                        lm.y,
                        lm.z,
                        lm.visibility,
                        0  # Non-seizure label
                    ])

            frame_count += 1

        cap.release()

pose.close()
print(f"All non-seizure landmarks saved to: {output_csv_path}")
