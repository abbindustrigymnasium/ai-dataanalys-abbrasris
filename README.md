## Captcha Solver

## Installation

```
pip install -r requirements.txt
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com/facebookresearch/detectron2.git"
```

## Användning

```python3
flask run
```

Demovärden:

| Key     | Value            | Type |
| ------- | ---------------- | ---- |
| label   | bicycle          | Text |
| shape_x | 3                | Text |
| shape_y | 3                | Text |
| image   | 2_bicycle_1.jfif | FIle |
