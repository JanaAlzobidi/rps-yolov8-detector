# Rock Paper Scissors Detector (YOLOv8)

An object detection model that finds hands in a photo and recognises the gesture: **rock**, **paper** or **scissors**. On top of the model we built a two-player game: the hand on the left is Player 1, the hand on the right is Player 2, and the app announces the winner.

> This README is being written. Each section below says which folder its content comes from.

## Team

| Member | Role | Folder |
|---|---|---|
| Jana | Repository, project README | this page |
| Raghad | Training: notebook, best model | [`1_training/`](1_training/) |
| Rawan | Deployment: Gradio app | [`2_deployment/`](2_deployment/) |
| Member 4 | Testing and evaluation on new images | [`3_evaluation/`](3_evaluation/) |

## Repository structure

```
README.md          project overview (this page)
1_training/        training notebook, best.pt, results of the experiments
2_deployment/      the Gradio app and how to run it
3_evaluation/      tests on new images and the answers to "what works, what fails"
```

## Problem

*Which problem we chose and why.*

## Dataset

*Classes, number of images, split, preprocessing and augmentation. Source: Roboflow and `1_training/`.*

## Training and experiments

*Model sizes and hyperparameters we compared, and which model we picked. Source: `1_training/`.*

## Results

*mAP, precision and recall of the best model, per class. Source: `1_training/` and `3_evaluation/`.*

## What went wrong and why

*Mistakes the model makes, and what we changed between dataset versions. Source: `3_evaluation/`.*

## Deployment

*How to run the game, with a screenshot or a link. Source: `2_deployment/`.*

## How to contribute

1. Work only inside **your own folder**, so uploads do not overwrite each other.
2. Upload on GitHub: open your folder, then **Add file > Upload files**, and write a short message saying what you added.
3. **Never upload a Roboflow API key.** In the notebook, use `api_key=getpass("Roboflow API key: ")` instead of writing the key.
