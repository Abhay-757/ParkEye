## **ParkEye**

🚗 Smart Parking Detection System

A lightweight computer vision system to detect empty vs occupied parking spots from video using classical ML.

🧠 Project Flow
Input Video
     ↓
Frame Extraction (OpenCV)
     ↓
Parking Slot Detection (Mask + Connected Components)
     ↓
Crop Each Slot
     ↓
Preprocessing (Resize 15×15 → Flatten)
     ↓
SVM Classifier (Empty / Not Empty)
     ↓
Draw Bounding Boxes (Green / Red)
     ↓
Display Available Spots


⚙️ Approach
Images resized to 15×15 and flattened
Model: scikit-learn SVM (with GridSearch)
Real-time processing using OpenCV
Optimized using frame differencing (only update changing spots)
🎯 Output
🟩 Green → Empty
🟥 Red → Occupied
Live counter of available slots
🔄 Alternative (Deep Learning)

Also experimented with TensorFlow / TFLite:


🚀 How to Use the Project

Follow these simple steps to run the parking detection system:

📥 1. Open the Project
Clone or download the repository
Open it in your IDE (VS Code / PyCharm)
🎥 2. Add Input Files
Place your parking video in the project folder
Place the corresponding mask image (same resolution as video)

👉 Example:

video_path = 'parking_1920_1080.mp4'
mask = 'mask_1920-1080.png'
🧠 3. Load Trained Model
Ensure Model.p (trained classifier) is present in the project directory
▶️ 4. Run the Code
python main.py
👀 5. View Output
A window will open showing:
🟩 Green → Empty spots
🟥 Red → Occupied spots
Total available parking count


Better accuracy (CNN)
Suitable for edge/mobile deployment
🚀 Key Idea

Instead of detecting cars directly →
👉 classify each parking slot, making it faster and practical for real systems.
