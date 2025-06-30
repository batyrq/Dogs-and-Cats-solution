# Dogs vs. Cats Image Classification

-----

This repository contains the Jupyter Notebook `dogs-and-cats.ipynb`, which provides a basic solution for the "Dogs vs. Cats Redux: Kernels Edition" Kaggle competition. The goal is to classify images as either a dog or a cat.

## Competition Details

  * **Kaggle Competition Link:** [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition)
  * **Evaluation Metric:** LogLoss
  * **Achieved Score:** 0.51441

## Project Overview

This project implements a simple Convolutional Neural Network (CNN) in PyTorch to classify images of dogs and cats.

The key steps in the solution are:

1.  **Data Extraction:**
      * Unzipping `train.zip` and `test.zip` to extract the image files into the working directory.
2.  **Data Preparation:**
      * **Label Encoding:** The image filenames (e.g., `dog.1234.jpg`, `cat.5678.jpg`) are used to extract labels (`dog` or `cat`), which are then numerically encoded using `sklearn.preprocessing.LabelEncoder`.
      * **Custom PyTorch `Dataset` (`CustomDataset` for training, `CustomDataset1` for testing):**
          * These classes handle loading images from the specified directory.
          * They extract the animal label from the filename for training images and convert it to its numerical representation.
          * For test images, they return the image and its original filename (ID).
      * **Image Transformations:** Images are resized to `(64, 64)` pixels and converted to PyTorch tensors using `torchvision.transforms.Compose`.
      * **`DataLoader`:** Used to efficiently load batches of images and labels/filenames for training and inference.
3.  **Model Definition (`CNN`):**
      * A custom CNN model is defined using `torch.nn.Module`.
      * It consists of a convolutional layer (`nn.Conv2d`) with ReLU activation and two `MaxPool2d` layers to downsample the feature maps.
      * A `nn.Flatten()` layer converts the 2D feature maps into a 1D vector.
      * Two fully connected layers (`nn.Linear`) with a ReLU activation in between are used for classification, outputting 2 values (one for dog, one for cat).
4.  **Training:**
      * The model is moved to the GPU (`cuda`) if available.
      * `nn.CrossEntropyLoss` is used as the loss function.
      * The `Adam` optimizer is used for training.
      * The model is trained for 10 epochs. During the last epoch, the accuracy on the training set is calculated.
5.  **Inference and Submission:**
      * The trained model is used to make predictions on the test dataset.
      * For each test image, the model outputs raw scores (logits) for "dog" and "cat".
      * `torch.argmax` is used to get the predicted class index (0 or 1).
      * The predicted class is then used to populate the `sample_submission.csv` file. **Note:** There appears to be an issue in the provided code snippet during the submission generation phase where `sample_submission.loc[int(filename)-1,'id'] = predictions[j]` is used. For LogLoss, probabilities are needed, and the `id` column should correspond to the image ID, while `label` should be the predicted probability of being a dog. The current implementation attempts to put predictions into the 'id' column and uses `answer[j].item()` which is not defined in the test loop, leading to an error and incorrect submission format.

## Setup and Running the Notebook

To run this notebook, you'll need a Kaggle environment or a local setup with the necessary libraries.

### Prerequisites

  * Python 3.x
  * `pandas`
  * `zipfile` (standard library)
  * `os` (standard library)
  * `PIL` (Pillow)
  * `torch`
  * `torchvision`
  * `numpy`
  * `matplotlib`
  * `scikit-learn`

### Installation

You can install the required Python packages using pip:

```bash
pip install pandas Pillow torch torchvision numpy matplotlib scikit-learn
```

### Running the Notebook

1.  **Download the data:** Download `train.zip`, `test.zip`, and `sample_submission.csv` from the Kaggle competition page and place them in your input directory (e.g., `/kaggle/input/dogs-vs-cats-redux-kernels-edition/` if on Kaggle).
2.  **Open the notebook:** Open `dogs-and-cats.ipynb` in a Jupyter environment (Jupyter Lab, Jupyter Notebook, Google Colab, or Kaggle Notebooks).
3.  **Run all cells:** Execute all cells in the notebook sequentially. The script will perform data extraction, model training, and attempt to generate a `submission.csv` file.

-----
