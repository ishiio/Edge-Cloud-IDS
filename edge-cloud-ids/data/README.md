# Included real research datasets

Both CSVs are bundled from user uploads. `manifest.json` records SHA-256 checksums.

- `edge_iiotset/ML-EdgeIIoT-dataset.csv`: 157,800 rows; 156,986 after exact deduplication; 15 classes.
- `ton_iot/train_test_network.csv`: 211,043 rows; 190,474 after exact deduplication; 10 classes.

Publisher references:
- Edge-IIoTset: https://ieee-dataport.org/8939
- TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets

Observe the original publishers' citation and usage requirements. Files were supplied by the user; their publisher provenance was not independently authenticated.

The supplied models are trained with seed 0. Validation selects the cloud model, calibration and threshold. Test results are independent implementation results, not a verified reproduction of the paper. Live flows replay the held-out test set; latency and INT8 remain simulated.

Run `python -m service.run`. To retrain: `python -m ids.train --dataset all`.
To produce a new five-seed study: `python -m ids.train --benchmark --seeds 5`.
The historical synthetic benchmark is archived and is not served as the real-data benchmark.
