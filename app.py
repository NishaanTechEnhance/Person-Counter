from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker  
import pandas as pd
import time
import os

app = Flask(__name__)
model = YOLO('yolov8n.pt')

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

tracker = Tracker()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    processed_frame_interval = 10
    resize_width = 640

    # Prepare for video output with the same codec as input
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_video_path = 'output_video.mp4'
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    id_positions = {}  # Dictionary to store the last known positions of detected IDs
    person_counts = []  # List to store person counts

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % processed_frame_interval != 0:
            # Draw existing boxes
            for id, (x3, y3, x4, y4) in id_positions.items():
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.putText(frame, str(int(id)), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, f"Live Person Count: {len(id_positions)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(frame)
            continue

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * resize_width / frame.shape[1])))

        results = model.predict(resized_frame)
        a = results[0].boxes.data
        
        px = pd.DataFrame(a).astype("float")
        list_of_boxes = []
        for index, row in px.iterrows():
            x1 = int(row[0] * frame.shape[1] / resize_width)
            y1 = int(row[1] * frame.shape[1] / resize_width)
            x2 = int(row[2] * frame.shape[1] / resize_width)
            y2 = int(row[3] * frame.shape[1] / resize_width)
            d = int(row[5])
            c = class_list[d]
            if c == "person":
                list_of_boxes.append([x1, y1, x2, y2])

        bbox_idx = tracker.update(list_of_boxes)
        current_ids = set()
        for bbox in bbox_idx:
            x3, y3, x4, y4, id = bbox
            current_ids.add(id)
            id_positions[id] = (x3, y3, x4, y4)

        # Remove IDs that are no longer detected
        id_positions = {id: pos for id, pos in id_positions.items() if id in current_ids}

        # Draw all boxes from the dictionary
        for id, (x3, y3, x4, y4) in id_positions.items():
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.putText(frame, str(int(id)), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        # Store the current number of people in the frame
        person_counts.append(len(id_positions))

        # Display the current number of people in the frame
        cv2.putText(frame, f"Live Person Count: {len(id_positions)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
    
    cap.release()
    out.release()
    # Calculate the average person count
    average_person_count = int(np.mean(person_counts)) if person_counts else 0
    return temp_video_path, average_person_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    upload_path = file.filename
    file.save(upload_path)
    
    with app.app_context():
        start_time = time.time()
        processed_video_path, average_person_count = process_video(upload_path)

        return render_template('index.html', 
                               processed_video=processed_video_path, 
                               average_person_count=average_person_count,
                               filename=os.path.basename(processed_video_path))

@app.route('/download/<filename>')
def download_file(filename):
    file_path = filename
    return send_file(file_path, as_attachment=True, download_name="output_video.mp4")

if __name__ == '__main__':
    app.run(debug=True)
