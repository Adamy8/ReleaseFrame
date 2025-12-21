# ReleaseFrame

Frame-by-frame baseball pitch release detection with a 2D CNN baseline and a CNN-TCN sequence model.

## Quick start
- Clone the repo:
```bash
git clone https://github.com/Adamy8/ReleaseFrame.git
cd ReleaseFrame
```
- Python 3.10+ recommended. Create a virtualenv if desired.
- Install deps: `pip install -r requirements.txt`.
- Download the `annotations/` folder (JSON labels + raw clips) from the project Google Drive into the repo root.

## Prepare data
1) Rename clips to sequential names that match the annotations:  
   `python processdata.py rename`
2) Build the training dataset structure (frames, labels, metadata):  
   `python processdata.py build-dataset`  
   Output defaults to `dataset/` with subfolders for images and labels.

## Train
#### 2D CNN baseline
- Train: `python train_cnn.py --dataset dataset --epochs 5 --batch-size 32 --lr 1e-4`

#### CNN-TCN
- (Optional) create fresh splits if you do not already have `splits/`:  
  `python train_cnn_tcn.py --dataset dataset --make-splits --seed 42`
- Train: `python train_cnn_tcn.py --dataset dataset --epochs 25 --batch-size 4 --device cuda`

## Inference
- 2D CNN: `python predict_release_2Dcnn.py --video path/to/input.mp4 --model best_model.pth`
- CNN-TCN: `python predict_release_tcn.py --video path/to/input.mp4 --model best_model.pth --min-frames 10`

## Samples
- `annotated_dodgers.mp4`: 2D CNN annotated output.
- `annotated_dodgers_tcn.mp4`: CNN-TCN prediction preview.
