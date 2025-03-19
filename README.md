<h1 align="center">Coffee Leaf Disease Prediction 🌱☕</h1>

<p align="center">
  <strong>Detect coffee leaf diseases with AI to protect yields.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.15-orange.svg" alt="TensorFlow 2.15"></a>
  <a href="https://github.com/yourusername/coffee_leaf_disease_cnn/actions"><img src="https://img.shields.io/github/last-commit/yourusername/coffee_leaf_disease_cnn.svg" alt="Last Commit"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
</p>

---

## 🌍 Why This Project?

Ethiopia 🇪🇹, the **10th largest coffee exporter** in the world, relies on coffee for **34% of its exports**. Yet, coffee leaf diseases like Rust and Phoma cause a devastating **30% yield loss annually**, threatening farmers’ livelihoods. Early visual identification is one of the best defenses, but it’s tricky—even for trained experts. This **Convolutional Neural Network (CNN)**, built from scratch, automates early detection, classifying diseases with ~80-90% accuracy to help coffee growers act fast and save their crops.

---

## ✨ Features

- 🚀 **Custom CNN**: Lightweight, tailored for coffee leaf classification.
- 🔍 **Multiclass Support**: Detects Healthy, Miner, Phoma, Rust (and more with tweaks).
- 🛠️ **User-Friendly**: Simple scripts for training and prediction.
- 🌐 **Open Source**: Free to use, modify, and share!

---

## 📂 Project Structure

```
coffee_leaf_disease_predictions/
├── data/                 
│   ├── train/             
│   └── test/              
├── models/                
│   └── my_coffee_leaf_model.h5  
├── main/                   
│   ├── data_loader.py     
│   ├── model.py           
│   ├── train.py          
│   └── predict.py         
├── README.md            
├── requirements.txt      
└── .gitignore             
```

---

## 🏁 Get Started

### 1. Clone the Repo
```bash
git clone https://github.com/felekekinfe/Coffee-leaf-disease-predictions.git
cd coffee_leaf_disease_cnn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your Dataset
Organize images in `data/`:
```
data/
  train/
    Rust/
    Healthy/
  test/
    Rust/
    Healthy/
```
- Aim for 1,000+ images for best results.

---

## 🎯 Usage

### Train the Model
```bash
python main/train.py
```
- 📈 Plots accuracy/loss curves.
- 💾 Saves to `models/my_coffee_leaf_model.h5`.

### Predict on an Image
```bash
python main/predict.py --img_path path/to/your/leaf.jpg
```
- 🖼️ Shows the image.
- ✅ Outputs prediction (e.g., "Rust - 87.34%").
- ✏️ Adjust `class_names` in `predict.py` to match your labels.

---

## 📊 Results

- **Accuracy**: ~80-90% with a solid dataset.
- **Classes**: Healthy, Miner, Phoma, Rust (customizable).
- **Impact**: Supports early detection to combat Ethiopia’s 30% yield loss.

<p align="center">
  <img src="https://via.placeholder.com/400x300.png?text=Prediction+Example" alt="Prediction Example" width="400"/>
  <br><em>Example: "Rust - 87.34%"</em>
</p>

---

## 🛠️ Requirements
Key dependencies (see `requirements.txt`):
- `tensorflow==2.15.0`
- `numpy==1.26.4`
- `matplotlib==3.8.3`

Install with:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Love coffee or AI? Join us!
1. Fork this repo 🍴.
2. Branch: `git checkout -b feature/your-idea`.
3. Commit: `git commit -m "Add feature"`.
4. Push: `git push origin feature/your-idea`.
5. Pull Request 📬.



---

## 📜 License
[MIT License](LICENSE) - Free to use and share.

---

## 🌟 Acknowledgments
- Inspired by Ethiopia’s coffee heritage and the fight against yield loss.
- Built with TensorFlow, Keras, and open-source enthusiasm by feleke kinfe

---

## 📬 Contact
Questions? Ideas? Open an [issue](https://github.com/felekekinfe/Coffee-leaf-disease-predictions/issues) or reach out to feleke kinfe on GitHub!

---

<p align="center">
  <em>“Empowering coffee farmers with AI”</em> ☕🌱
</p>



