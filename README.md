# Deep Learning for Computer Vision — Labs

## Run the notebooks

Install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

Open the notebook in Jupyter or VS Code, select that Python environment, and
run all cells from the beginning. Keep the repository folder structure intact:
the notebooks import their local implementation and the shared `dlcv2026`
package. Save the notebook after running it so the results remain visible.

| Lab | Notebook | Implementation |
| --- | --- | --- |
| 01 | `LAB01/Practice_01.ipynb` | `LAB01/pytorch101.py` |
| 02 | `LAB02/knn.ipynb` | `LAB02/knn.py` |
| 03 | `LAB03/knn.ipynb` | `LAB03/knn.py` |
| 04 | `LAB04/lab04_svm_loss.ipynb` | `LAB04/linear_classifier.py` |
| 05 | `LAB05/lab05_svm_training.ipynb` | `LAB05/linear_classifier.py` |

LAB02 and LAB03 download the real CIFAR-10 dataset on their first run and reuse
it from the ignored `data/` directory. To use an existing dataset cache, set
the `DLCV_DATA_DIR` environment variable to its directory. LAB04 runs on CPU
with `reg=0.0` and checks its analytic gradient with finite differences.

LAB05 implements the vectorized regularized SVM loss and gradient. Its notebook
includes executed outputs, training and regularization experiments, and wrap-up
answers. The local `LAB05/dlcv2026` utilities keep it self-contained. See
[LAB05/README.md](LAB05/README.md) for run commands and validation results.
