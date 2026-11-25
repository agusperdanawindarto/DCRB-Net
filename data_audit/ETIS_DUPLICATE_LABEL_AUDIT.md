# DCRB-Net ETIS-Larib — Exact-image duplicate / differing-mask audit

During SHA-256 validation of 196 external-test image–mask pairs from the Zenodo benchmark, three instances were identified where **two image files are byte-for-byte identical** but their provided mask files are not.

| Exact-image duplicate frame IDs | Same image bytes | Same mask bytes | Binary-mask Dice between annotations |
|---|---|---|---:|
| 20.png / 22.png | Yes | No | 0.8861 |
| 127.png / 131.png | Yes | No | 0.9623 |
| 172.png / 175.png | Yes | No | 0.9598 |

The mask overlaps were calculated after grayscale conversion and thresholding at >127. These overlaps indicate similar but non-identical annotations. No correction has been applied to either masks or images. The file count remains **196 pairs**, containing **193 distinct SHA-256 image byte streams**.

**Research handling:** Retain all provided benchmark samples for the principal published-protocol comparison. Flag these samples and, if appropriate, add a separate sensitivity calculation excluding duplicates or selecting one annotation per identical image. Never let ETIS images influence model training or validation.

**Provenance:** Zenodo record https://zenodo.org/records/5579392, `TestDataset/TestDataset/ETIS-LaribPolypDB`.
