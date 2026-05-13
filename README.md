# Emotion Detection

Deep learning notebook for facial emotion analysis. The project trains and evaluates two TensorFlow/Keras models:

- A facial keypoint detector that predicts 15 facial landmark points, represented as 30 `(x, y)` coordinates.
- A facial expression classifier that predicts one of five emotions: `anger`, `disgust`, `sad`, `happiness`, or `surprise`.

The notebook also combines both models so an input face image can be annotated with predicted facial landmarks and an emotion label, then shows how to export and serve the models with TensorFlow Serving.

## Repository Contents

```text
.
+-- Emotion_Detector.ipynb   # Main notebook: preprocessing, training, evaluation, deployment
+-- LICENSE
+-- README.md
```

## Dataset

The notebook expects the dataset files to be available in Google Drive using the original Colab paths:

```text
/content/drive/My Drive/Colab Notebooks/P96-Section-2-Emotion-AI/Emotion+AI+Dataset/Emotion AI Dataset/
```

Expected files include:

- `data.csv` for facial keypoint detection.
- `icml_face_data.csv` for facial expression classification.
- `detection.json` and `weights_keypoint.hdf5` for loading the trained keypoint model.
- `emotion.json` and `weights_emotions.hdf5` for loading the trained emotion model.

These dataset and weight files are not included in this repository. If you store them somewhere else, update the file paths in `Emotion_Detector.ipynb`.

## Setup

This notebook was written for Google Colab. The easiest way to run it is:

1. Upload or clone this repository into Google Drive.
2. Place the dataset folder at the path shown above, or update the notebook paths.
3. Open `Emotion_Detector.ipynb` in Google Colab.
4. Use a GPU runtime:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Install the main Python dependencies if they are not already available:

```bash
pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn opencv-python pillow requests
```

The TensorFlow Serving section additionally installs `tensorflow-model-server` inside the Colab runtime with `apt-get`.

## Notebook Workflow

### 1. Facial Keypoint Detection

The notebook loads `data.csv`, where each image is stored as a space-separated pixel string and each row includes 30 facial keypoint coordinate values.

Processing steps:

- Convert image strings into `96 x 96` grayscale arrays.
- Visualize faces with landmark coordinates.
- Augment the dataset with horizontal flips and brightness changes.
- Normalize image pixels to `[0, 1]`.
- Split into train and test sets.
- Train a custom residual CNN for coordinate regression.

Keypoint dataset details from the notebook:

- Original rows: `2,140`
- Original image size: `96 x 96`
- Augmented rows: `6,420`
- Model input shape: `(96, 96, 1)`
- Target shape: `30` facial coordinate values
- Train split: `5,136` images
- Test split: `1,284` images

### 2. Facial Expression Classification

The notebook loads `icml_face_data.csv`, where each row contains an emotion label and face pixels.

Processing steps:

- Convert pixel strings into arrays.
- Resize images from `48 x 48` to `96 x 96`.
- One-hot encode emotion labels.
- Split into train, validation, and test sets.
- Apply image augmentation with rotation, shifting, shear, zoom, flips, and brightness changes.
- Train a residual CNN classifier with a softmax output.

Emotion labels:

```python
{
    0: "anger",
    1: "disgust",
    2: "sad",
    3: "happiness",
    4: "surprise",
}
```

Emotion dataset details from the notebook:

- Total rows: `24,568`
- Null values: `0`
- Train split: `22,111` images
- Test split: `1,229` images
- Model input shape: `(96, 96, 1)`
- Number of classes: `5`

Class distribution:

| Emotion | Label | Count |
| --- | ---: | ---: |
| happiness | 3 | 8,989 |
| sad | 2 | 6,077 |
| anger | 0 | 4,953 |
| surprise | 4 | 4,002 |
| disgust | 1 | 547 |

### 3. Combined Prediction

The notebook defines a `predict()` function that:

1. Uses the keypoint model to predict landmark coordinates.
2. Uses the emotion model to predict the emotion class.
3. Combines both outputs into one dataframe.
4. Visualizes test images with predicted keypoints and emotion labels.

### 4. Deployment

The final notebook section exports both trained models using `tf.saved_model.save()` and serves them with TensorFlow Serving:

- Keypoint model REST port: `4500`
- Emotion model REST port: `4000`

Example endpoints used in the notebook:

```text
http://localhost:4500/v1/models/keypoint_model/versions/1:predict
http://localhost:4000/v1/models/emotion_model/versions/1:predict
```

## Results

### Facial Keypoint Model

The keypoint model was trained for 2 epochs in the notebook.

Training output:

| Epoch | Loss | Accuracy | Validation Loss | Validation Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 611.4508 | 0.4801 | 920.2023 | 0.6926 |
| 2 | 113.1486 | 0.6048 | 354.7619 | 0.6926 |

Evaluation output from the loaded trained keypoint model:

| Metric | Value |
| --- | ---: |
| Test loss | 4.9989 |
| Test accuracy | 0.8489 |

### Facial Expression Model

The notebook builds, trains, and evaluates the emotion classifier, but the saved notebook output does not include the final value printed by:

```python
score = model_2_emotion.evaluate(X_Test, y_Test)
print("Test Accuracy: {}".format(score[1]))
```

Run the evaluation cells in `Emotion_Detector.ipynb` after loading `emotion.json` and `weights_emotions.hdf5` to reproduce the final test accuracy.

## Generated Model Files

When the training cells are run, the notebook writes model artifacts such as:

- `FacialKeyPoints_weights.hdf5`
- `FacialKeyPoints-model.json`
- `FacialExpression_weights.hdf5`
- `FacialExpression-model.json`

These generated files are ignored unless you explicitly add them to the repository.

## Notes

- The notebook uses absolute Google Drive paths, so update paths before running locally.
- The Colab-specific import `google.colab.patches.cv2_imshow` should be removed or replaced if running outside Colab.
- Some TensorFlow Serving commands use Linux shell syntax and are intended for Colab or another Linux environment.
