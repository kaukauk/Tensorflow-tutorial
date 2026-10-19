# Tensorflow-tutorial

My first deep-learning project (2021). I trained convolutional neural networks with TensorFlow/Keras, first on MNIST digits, then on German traffic signs and a fish species dataset from Kaggle.

## What it does

- `mnist_ex.py`: trains a small CNN on MNIST, built three ways (Sequential, Functional API and subclassed `keras.Model`) in `deeplearning_models.py`
- `gtsrb_model.py`: trains and evaluates a 43-class traffic-sign classifier on the GTSRB dataset
- `fish_testing.py`: trains a 9-class fish classifier using the same architecture
- `street_predictor.py`: loads a saved model and predicts the class of a single image
- `deep_utils.py`: helpers for splitting a dataset into train/validation folders, sorting the GTSRB test set into class folders, building `ImageDataGenerator` pipelines, and checkpoint/early-stopping callbacks
- Trained models are saved in `Street_sign_model/` and `Fish_model/` (TensorFlow SavedModel format)

## Datasets

- Street signs (GTSRB): https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- Fish: https://www.kaggle.com/crowww/a-large-scale-fish-dataset

The datasets are not included. Download them from Kaggle.

## Tech

Python 3.8 (Miniconda), TensorFlow 2 / Keras, NumPy, scikit-learn, Matplotlib

## Running it

```bash
pip install tensorflow numpy scikit-learn matplotlib
python mnist_ex.py
```

`mnist_ex.py` downloads MNIST automatically. The GTSRB, fish and prediction scripts use hard-coded absolute dataset and model paths, so update those paths in each script before running `gtsrb_model.py`, `fish_testing.py` or `street_predictor.py`.

## Notes

This is a learning project that follows a TensorFlow/Keras tutorial course (the MNIST and GTSRB parts). Applying the same approach to the fish dataset was my own extension.
