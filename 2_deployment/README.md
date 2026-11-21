# 2_deployment

Gradio app for the Rock Paper Scissors detector (two players, one image, the model decides the winner).

## Contents
- `app.py` - the Gradio app with home, how it works, about and game pages plus a settings menu (loads `best.pt`, detects 2 hands, judges the winner, keeps score)
- `best.pt` - trained YOLOv8 weights (classes: paper, rock, scissors)
- `requirements.txt` - dependencies
- `Homepage.png` - screenshot of the home page
- `screenshotGame.png` - screenshot of the game running with a prediction

## Screenshot
### Home page
![Home page](Homepage.png)

### Game
![Game screenshot](screenshotGame.png)

## Run locally
```
pip install -r requirements.txt
python app.py
```

## How it works
1. The image goes through YOLOv8.
2. **Game Mode** (default) uses a fixed confidence of 0.5. **Advanced Settings** (side menu: Settings button on the game page) shows a slider (0.1 to 0.9) to change it.
3. The 2 most confident detections are kept and sorted left to right: left = Player 1, right = Player 2.
4. Rock beats scissors, scissors beats paper, paper beats rock. Same gesture = tie.
