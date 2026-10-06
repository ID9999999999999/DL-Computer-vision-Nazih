# Lab 05 — Vectorized SVM Training

Multiclass hinge loss, its manual gradient, and minibatch SGD on the course's
three-class toy dataset. The notebook contains the experiments and answers.

## Run

From the repository root:

```bash
python -m pip install -r LAB05/requirements.txt
python LAB05/run_notebook.py
python -m unittest discover -s LAB05 -v
```

The notebook also runs in Jupyter or VS Code. Keep `linear_classifier.py` and
`dlcv2026` beside it, select the Python environment, and run all cells in order.
No dataset download is needed.

## Results

CPU, float64, Python 3.12.14, PyTorch 2.14.1+cpu.
All 10 code cells and 11 tests passed.

| Check | Result |
| --- | --- |
| Fixture loss, `reg=0` / `reg=0.1` | 2.0 / 2.4 |
| Naive/vectorized loss difference | 4.44e-16 |
| Maximum gradient difference | 2.78e-16 |
| Vectorized finite-difference relative error | 5.99e-10 |
| Training accuracy, 360 examples | 99.72% |
| Validation accuracy, 120 examples | 100.00% |
| Weight norm, `reg=0.001` / `reg=1.0` | 1.5777 / 0.5398 |

These accuracies concern synthetic data. Timings in the notebook use warmed-up,
single-thread CPU measurements and depend on the machine.

The tests cover equivalence with the loop reference, an independent autograd
gradient, zero margins, invalid inputs, and reproducible training.

## Source

Adapted from [course Lab05](https://github.com/naziherrahel/Deep-Learning-for-Computer-Vision/tree/main/Labs/Lab05).
The loop reference, training pipeline, and dataset come from the course.
