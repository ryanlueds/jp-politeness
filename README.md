# Dataset
We used the [JGLUE](https://github.com/yahoojapan/JGLUE?tab=readme-ov-file) dataset. Specifically, we used a subset called JCommonsenseQA.

# Files
```
.
├── data
│   └── jcommonsense
│       └── rewritten_dataset.json      # dataset variations
├── environment.yml                     # conda environment dependencies
├── README.md
└── src
    ├── analyze_dataset_complexity.py   # analysis over rewritten dataset
    ├── evaluator.py                    # evaluate model on rewritten dataset
    └── rewriter.py                     # generate rewritten dataset
```
