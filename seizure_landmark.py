import cv2
import os
import csv
import mediapipe as mp

# Folder with seizure videos
video_folder = './seizure_class'

# Output merged CSV path
output_csv_path = './seizure_pose_landmarks_all.csv'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the CSV file for writing all seizure data
with open(output_csv_path, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    
    # Write the header with seizure label
    csv_writer.writerow([
        'video_name', 'frame', 'landmark_index', 
        'x', 'y', 'z', 'visibility', 'label'  # 'label' added here
    ])

    # Process all videos in the folder
    for video_file in os.listdir(video_folder):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

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
                        1  # Seizure label
                    ])

            frame_count += 1

        cap.release()

pose.close()
print(f"All seizure video landmarks saved to: {output_csv_path}")
