# EEG input contract

The trainer reads averaged EEG arrays without changing preprocessing:

| File | Shape | Row order |
| --- | --- | --- |
| `averaged_eeg_training.npy` | `[16540, 63, 250]` | Concept, then image within concept |
| `averaged_eeg_test.npy` | `[200, 63, 250]` | Test image order from metadata |

The workspace converter, `preprocess_things_eeg2_nice.py`, applies a 0.1-100 Hz
software filter before epoching, baseline correction with the preceding
200 ms, resampling to 250 Hz, and training-derived session MVNN (spatial
whitening). It averages four training repetitions and 80 test repetitions.
Check each subject's `metadata.json` to determine which preprocessing produced
an existing dataset; editing converter settings does not change saved arrays.

The trainer optionally smooths a deterministic subset of channels. Its mask
uses subject, image, channel, and seed to stay fixed across epochs without
another full EEG copy. Input files are never overwritten by training.
