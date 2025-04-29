# Cleanser

## 🖼️ Overview

**Cleanser** is a Python-based application designed for applying various image processing techniques. It provides functionalities such as grayscale conversion, histogram equalization, noise addition, filtering, edge detection, thresholding, and hybrid image creation. The application features a graphical user interface (GUI) for user-friendly interaction.

## ✨ Features

- **Grayscale Conversion**: Transform RGB images to grayscale.
 <div align="center">
   <img src="https://github.com/Eagle-E-Y-E/Cleanser/blob/main/report/grayscale.png" width="300" alt="grayscale.png">
</div>

- **Histogram Equalization**: Enhance image contrast using histogram equalization.
  <div align="center">
   <img src="https://github.com/Eagle-E-Y-E/Cleanser/blob/main/report/main.png" width="800" alt="equalization">
</div>

- **Noise Addition**: Introduce different types of noise to images for testing.
<div align="center">
   <img src="https://github.com/Eagle-E-Y-E/Cleanser/blob/main/report/gaussian.png" width="200" alt="equalization">
</div>

- **Filtering Techniques**: Apply various filters to images.

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/Eagle-E-Y-E/Cleanser/blob/main/report/avg_filter.png" width="200"><br>
      <small>Average Filter</small>
    </td>
    <td align="center">
      <img src="https://github.com/Eagle-E-Y-E/Cleanser/blob/main/report/gaussian_filter.png" width="200"><br>
      <small>Gaussian Filter</small>
    </td>
    <td align="center">
      <img src="https://github.com/Eagle-E-Y-E/Cleanser/blob/main/report/med_filter.png" width="200"><br>
      <small>Median Filter</small>
    </td>
  </tr>
</table>



- **Edge Detection**: Detect edges within images.
- **Thresholding**: Apply thresholding techniques to segment images.
- **Hybrid Image Creation**: Combine images to create hybrid visuals.
- **Graphical User Interface**: Interact with the application through a GUI.

## 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Eagle-E-Y-E/Task1-Image-filtration.git
   cd Task1-Image-filtration
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure that `requirements.txt` is present in the repository with all necessary dependencies listed.*

## 🚀 Usage

1. **Run the Application**:

   ```bash
   python main.py
   ```

2. **Using the GUI**:

   - Load an image using the GUI.
   - Select the desired image processing operation.
   - View and save the processed image.

## 📁 File Structure

- `main.py`: Entry point of the application.
- `ui_2.ui`: Qt Designer file for the GUI layout.
- `ui_handler.py`: Handles GUI interactions and events.
- `RGB2GRAY.py`: Contains functions for grayscale conversion.
- `Histogram_Equalization.py`: Implements histogram equalization.
- `add_noise.py`: Functions to add noise to images.
- `apply_filter.py`: Applies various filters to images.
- `EdgeDetection.py`: Implements edge detection algorithms.
- `Thresholding.py`: Contains thresholding techniques.
- `Hybrid.py`: Functions to create hybrid images.
- `RGB_Hist.py`: Generates RGB histograms.
- `Freq_filters.py`: Applies frequency domain filters.
- `images/`: Directory containing sample images.
- `report/`: Contains project reports and documentation.
## 📌 Future Enhancements

- Implement additional image processing techniques.
- Enhance the GUI with more features and better user experience.
- Optimize performance for processing large images.
- Add support for batch processing of images.
