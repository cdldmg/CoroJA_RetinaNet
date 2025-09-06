# CoroJA_RetinaNet
### source code for the paper:CoroJA_RetinaNet: A Multiscale Attention-Guided Framework for Automated Coronary Plaque Detection in CTA Images
### Manuscript ID: IEEE LATAM Submission ID: 9889 Authors:
- Xuan Nie
- Teng Li
- Yinan Yuan
- Zichen Yan
- Yiwen Liu
- Guangpu Zhou
- Bosong Chai
### Affiliation:
- Northwestern Polytechnical University
- Zhejiang University

---

###  Project Structure
├── backbone/ # Backbone network implementation  
├── network_files/ # Model architecture definitions  
├── train_utils/ # Training utilities   
├── README.md # Project documentation  
├── draw_box_utils.py # Utilities for drawing predicted boxes  
├── my_dataset.py # Dataset definition  
├── plot_curve.py # Plot training/validation curves  
├── predict.py # Script for single image prediction  
├── requirements.txt # Dependency list  
├── train.py # Training script  
├── transforms.py # Data augmentation & preprocessing  
├── validation.py # Validation script  

---

### Training
python train.py --epochs 50 --batch-size 8 --data-path ./dataset

---

### Validation
python validation.py --data-path ./dataset --weights ./save_weights/best_model.pth

---

### Inference
python predict.py --weights ./save_weights/best_model.pth --img ./test.jpg

---

### Visualization
python plot_curve.py

