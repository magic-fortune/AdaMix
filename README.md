# AdaMix:Adaptive CutMix Strategy for Semi-Supervised Medical Image Segmentation with Confidence-Based Region Exchange
## Our frame:
![avatar](./framework.png)

## About our code
### DataSet
Data could be got at *[ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC)* and *[promise12](https://promise12.grand-challenge.org/Download/)*.

### Run
'''
python train_ACDC_AdaMix.py
python train_promise12_AdaMix.py
'''

### Outcome
Our results are as follows:
#### ACDC
![avatar](./compare.png)

#### PROMISE12
| Method                 | Reference         | #Labeled   | #Unlabeled | DSC↑    | ASD↓    |
|------------------------|-------------------|------------|------------|---------|---------|
| U-Net                 |                   | 7 (20%)    | 0          | 60.88   | 13.87   |
| U-Net                 |                   | 35 (All)   | 0          | 84.76   | 1.58    |
| CCT | (CVPR'20)        | 7 (20%)    | 28 (80%)   | 71.43   | 16.61   |
| URPC      | (MedIA'22)       | 7 (20%)    | 28 (80%)   | 63.23   | 4.33    |
| SS-Net    | (MICCAI'22)      | 7 (20%)    | 28 (80%)   | 62.31   | 4.36    |
| SLC-Net | (MICCAI'22)    | 7 (20%)    | 28 (80%)   | 68.31   | 4.69    |
| SCP-Net | (CVPR'23) | 7 (20%)    | 28 (80%)   | 77.06   | 3.52    |
| ABD| (CVPR'24)     | 7 (20%)    | 28 (80%)   | 82.06   | 1.33    |
| **Ours**              |                   | 7 (20%)    | 28 (80%)   | **82.78** | **1.12** |
| ABD| (CVPR'24)     | 3 (10%)    | 32 (90%)   | 81.81   | 1.46    |
| **Ours**              |                   | 3 (10%)    | 32 (90%)   | **82.13** | **1.09** |


## Acknowledgement
We extend our heartfelt gratitude to ABD *[Markdown Guide](https://github.com/chy-upc/ABD)* for providing the code, which has been invaluable to our work.

