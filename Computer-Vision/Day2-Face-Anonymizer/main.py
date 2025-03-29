# Flow of the project: Read Image/Video --> Detect Faces --> Blur Faces --> Save Output

import cv2
import mediapipe as mp  # library to detect faces
import os
import argparse

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x1 + w, W)
            y2 = min(y1 + h, H)

            # Apply blur if region is valid
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], (30, 30))

    return img


# CLI Args
args = argparse.ArgumentParser()
args.add_argument("--mode", default="webcam")  # "image" or "video" or "webcam"
args.add_argument("--filePath", default=None)  # Incase of webcam specify the default value as None
args = args.parse_args()

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode == "image":
        img = cv2.imread(args.filePath)
        if img is None:
            print("❌ Could not read image.")
            exit()
        img = process_img(img, face_detection)
        cv2.imwrite(os.path.join(output_dir, "output.png"), img)
        print("✅ Saved image output to:", os.path.join(output_dir, "output.png"))

    elif args.mode == "video":
        if not os.path.exists(args.filePath):
            print("❌ File does not exist:", args.filePath)
            exit()

        print("📂 Found video file:", args.filePath)
        cap = cv2.VideoCapture(args.filePath)

        if not cap.isOpened():
            print("❌ Failed to open video.")
            exit()

        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to read first frame.")
            cap.release()
            exit()

        height, width = frame.shape[:2]

        output_path = os.path.join(output_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'MP4V' or 'XVID' # four character code to use while saving the video
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while ret:
            processed = process_img(frame, face_detection)
            out.write(processed)
            ret, frame = cap.read()
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"🧠 Processed {frame_count} frames...")

        cap.release()
        out.release()
        print("✅ Video saved to:", output_path)
      
    elif args.mode ==  "webcam":
        
        ## Opening the camera and reading the frames
        cap = cv2.VideoCapture(0)
        
        # Reading the frames from the captured video
        ret, frame = cap.read()
        
        while ret:
            
            # Processing the each fram chunk and blurring the detected face
            frame = process_img(frame,face_detection)
            
            cv2.imshow("frame",frame)
            cv2.waitKey(25)
            
            ret, frame = cap.read()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        
        cap.release()
            
            
            
            
            
    
