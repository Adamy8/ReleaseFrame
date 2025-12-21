# ReleaseFrame

## Run this project
### 2D CNN:
  (0. use python venv, pip install -r requirements.txt)
  1. Download the annotations/ folder from Google drive.
  2. python processdata.py rename     # to change name into readable video###.mp4
  3. python processdata.py build-dataset    # to put data into the structure for training
  4. python train_cnn.py --dataset dataset --epochs 5 --batch-size 32 --lr 1e-4      # train the 2D CNN
  5. python predit_release_2DCNN.py --video path/to/input.mp4    # to use the model
### CNN-TCN
  - same step 0-3
  \# to split dataset into random train/val/test, optional if you have splits/ already
  4. python train_cnn_tcn.py --dataset dataset --make-splits --seed 42     
  \# train CNN-TCN, use nohup for no interrupting
  5. python train_cnn_tcn.py --dataset dataset --epochs 25 --batch-size 4 --device cuda
  6. python predict_release_tcn.py --video dodgers.mp4 --model best_model.pth --min-frames 10

## Samples
- `annotated_dodgers.mp4`: sample 2D CNN annotated output video.
- `annotated_dodgers_tcn.mp4`: sample CNN-TCN prediction

