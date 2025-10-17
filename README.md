# fire-detector
Real-time fire detection system using TensorFlow, OpenCV, and a CNN model trained on Kaggle’s Fire Dataset.
Fire Detector – Real-Time Fire Detection Using Deep Learning  

# Overview  
This project uses a **Convolutional Neural Network (CNN)** trained on the Kaggle Fire Dataset to detect **fire in real-time from a webcam feed**.  
Built with **TensorFlow, Keras, and OpenCV**, it combines image classification with motion and color analysis to reduce false positives.  

##  Features  
- Real-time fire detection via webcam  
- Motion + color gating to reduce false alarms  
- CNN trained using TensorFlow/Keras  
- Visualization overlay with prediction confidence  
- Adjustable detection thresholds  

##  Tech Stack  
Python 3.11+, TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn, scikit-learn, TQDM  

##  Project Structure  
```
fire-detector/
│
├── models/
│   └── fire_cnn.keras
│
├── dataset/
│   ├── fire_images/
│   └── non_fire_images/
│
├── fire_detector.py
├── train_model.py
├── requirements.txt
└── README.md
```

##  Installation  
```
git clone https://github.com/sidelectron/fire-detector.git
cd fire-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

##  Running the Model  
### Training (optional)
```
python train_model.py
```
### Real-time Detection
```
python fire_detector.py
```

Press `q` to quit.  

##  Model Logic Summary  
1. CNN classifier → predicts fire probability  
2. Motion gate → ensures frame activity  
3. Color filter (HSV) → validates flame-like color regions  

Fire is confirmed only if **all three** conditions hold.  

##  Example Output  
| State | Example |
|:------|:---------|
|  Fire | FIRE 0.84 (red box) |
|  No Fire | NO FIRE 0.12 (green box) |

## License  
MIT License  

##  Author  
**Siddhant Khairnar (@sidelectron)**  
Applied AI | Stevens Institute of Technology

