
# Waste Classification Project

## Features
- Automatically classifies images of waste into **degradable** and **non-degradable** categories.  
- Fast and accurate predictions using a trained machine learning model.  
- Supports image preprocessing for better classification accuracy.  
- Easy to extend for additional waste categories.  

## Technologies Used
- **Python** – programming language for scripts and model implementation.  
- **TensorFlow / Keras** – building and training the classification model.  
- **NumPy & Matplotlib** – data handling and visualization.  
- **Jupyter Notebook / VS Code** – for development and testing.  

## Usage
**1. Clone the repository:**
```bash
git clone https://github.com/dhivya-shreetha-s/Waste_Classification.git
````

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the prediction script:**

```bash
python predict.py --image path_to_image
```

**4. The model will output the **waste category** of the given image.**

## File Structure

```bash
Waste_Classification/
│
├── model/                  # Trained ML model (.h5)
├── dataset/                # Images for training/testing
├── scripts/                # Python scripts for prediction and preprocessing
├── README.md               # Project information
└── requirements.txt        # Python dependencies
```

## Notes

1. Ensure your images are clear for better prediction accuracy.
2. The current model classifies only **degradable** and **non-degradable** waste.
3. For updates or contributions, please create a new branch and raise a pull request.

This version:  
- Fixes all markdown formatting issues.  
- Properly closes code blocks.  
- Uses consistent headings, numbering, and indentation.  
