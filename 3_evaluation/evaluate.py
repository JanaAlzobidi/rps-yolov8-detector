import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL = "best.pt"
IMAGES = "new_images"
OUT = "predictions"
CONF = 0.5

model = YOLO(MODEL)
os.makedirs(OUT, exist_ok=True)

classes = ["paper", "rock", "scissors", "background"]
matrix = [[0] * 4 for _ in range(4)]
rows = []

for name in sorted(os.listdir(IMAGES)):
    expected = name.split("_")[1].split("+")

    result = model.predict(os.path.join(IMAGES, name), conf=CONF, verbose=False)[0]
    result.save(filename=os.path.join(OUT, name))

    boxes = sorted(result.boxes, key=lambda b: float(b.xyxy[0][0]))
    detected = [model.names[int(b.cls)] for b in boxes]
    scores = [f"{model.names[int(b.cls)]} ({float(b.conf):.2f})" for b in boxes]

    exp_left, det_left = expected.copy(), detected.copy()
    for g in expected:
        if g in det_left:
            matrix[classes.index(g)][classes.index(g)] += 1
            exp_left.remove(g)
            det_left.remove(g)
    for e, d in zip(exp_left, det_left):          # wrong label
        matrix[classes.index(e)][classes.index(d)] += 1
    for e in exp_left[len(det_left):]:            # missed hand
        matrix[classes.index(e)][3] += 1
    for d in det_left[len(exp_left):]:            # extra detection
        matrix[3][classes.index(d)] += 1

    correct = len(exp_left) + len(det_left) == 0
    hits = len(expected) - len(exp_left)
    status = "Yes" if correct else ("Partly" if hits > 0 else "No")
    rows.append((name, " + ".join(expected), " + ".join(scores) or "nothing", status))

table = "| Photo | Expected | Detected | Correct? |\n|---|---|---|---|\n"
for r in rows:
    table += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n"
print(table)
with open(os.path.join(OUT, "results.md"), "w") as f:
    f.write(table)

# 6. Draw the confusion matrix
plt.imshow(matrix, cmap="Blues")
plt.xticks(range(4), classes)
plt.yticks(range(4), classes)
plt.xlabel("Detected")
plt.ylabel("Expected")
plt.title(f"Confusion matrix (conf = {CONF})")
for i in range(4):
    for j in range(4):
        plt.text(j, i, matrix[i][j], ha="center", va="center")
plt.savefig("confusion_matrix_conf05.png", dpi=150, bbox_inches="tight")
print("Done. Saved predictions/ and confusion_matrix_conf05.png")
