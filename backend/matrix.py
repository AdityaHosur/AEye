# Plot confusion matrix with matplotlib
import numpy as np
import matplotlib.pyplot as plt

cm = np.array([[219, 6],
               [33, 1512]])
labels = [["TP=219", "FN=6"], ["FP=33", "TN=1512"]]

fig, ax = plt.subplots(figsize=(4.5, 4.5))
im = ax.imshow(cm, cmap="Blues")

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0,1]); ax.set_xticklabels(["Copied","Distinct"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Copied","Distinct"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i][j], ha="center", va="center", color="black", fontsize=10)

plt.title("Confusion Matrix (τ = 0.45)")
plt.tight_layout()
plt.show()