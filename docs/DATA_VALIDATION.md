# Data Validation

The preprocessing pipeline uses one global AAMI mapping:

| ID | Class |
|---:|:---|
| 0 | N |
| 1 | S |
| 2 | V |
| 3 | F |
| 4 | Q |

The mapping is defined once in `src/data/labels.py` and must not be inferred independently from train, validation, or test data.

Before training, processed arrays should satisfy these invariants:

- equal sample counts for `x` and `y`;
- fixed beat length of 256 samples by default;
- finite signal values (no NaN or infinity);
- integer class IDs in `[0, 4]`;
- no overlap of record identifiers between splits.

Record-level disjointness is critical because multiple beats from one ECG recording are highly correlated. A beat-level random split can produce overly optimistic evaluation results through leakage.
