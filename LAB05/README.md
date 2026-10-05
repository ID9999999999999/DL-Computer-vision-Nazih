# Lab 05 — Vectorized SVM Training

The regularized multiclass SVM loss and its analytic gradient are implemented
without Python loops in `linear_classifier.py`. The notebook contains the
completed experiments, saved outputs and figures, and answers to all three
wrap-up questions.

## Run

```bash
python -m pip install -r LAB05/requirements.txt
```

Open `LAB05/lab05_svm_training.ipynb` in Jupyter or VS Code, select the installed
Python environment, and run all cells from the beginning. Keep the local
`dlcv2026` folder next to the notebook. The supplied dataset requires no download.

Alternatively, execute the notebook and save its outputs using IPython:

```bash
python LAB05/run_notebook.py
```

Run the additional checks:

```bash
cd LAB05
python -m unittest -v test_linear_classifier.py
```

## Results

Executed on CPU with Python 3.12.14 and PyTorch 2.14.1, using float64.
All 10 notebook code cells and all 6 additional tests passed.

| Check | Observed result |
| --- | --- |
| Fixture data loss | 2.0 |
| Fixture loss with `reg=0.1` | 2.4 |
| Naive/vectorized loss absolute difference | 4.44e-16 |
| Maximum gradient absolute difference | 2.78e-16 |
| Vectorized finite-difference maximum relative error | 5.99e-10 |
| Training accuracy (360 examples) | 99.72% |
| Validation accuracy (120 examples) | 100.00% |
| Weight norm, `reg=0.001` | 1.5777 |
| Weight norm, `reg=1.0` | 0.5398 |

The accuracy is measured on the supplied three-class synthetic dataset.
Timing depends on the machine and is reported in the notebook.

The tests also compare float32/float64 results across random fixtures, compare
the manual derivative with autograd, check zero margins and a single-class
case, reject an empty minibatch, and verify deterministic training.

## Source

Based on the [course Lab05](https://github.com/naziherrahel/Deep-Learning-for-Computer-Vision/tree/main/Labs/Lab05).
The naive reference, training helpers, and teaching utilities are supplied by
the course. The vectorized implementation, answers, and additional checks
complete the required lab.
