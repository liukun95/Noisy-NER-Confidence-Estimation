![title](doc/title.png)

Kun Liu*, Yao Fu*, Chuanqi Tan, Mosha Chen, Ningyu Zhang, Songfang Huang, Sheng Gao. Noisy-Labeled NER with Confidence Estimation. NAACL 2021. [[arxiv](https://arxiv.org/abs/2104.04318)]

## Requirements
```bash
pip install -r requirements.txt

```
## Data
The format of datasets includes three columns, the first column is word, the second column is noisy labels and the third column is gold labels. For datasets without golden labels, you could set the third column the same as the second column. We provide the CoNLL 2003 English with recall 0.5 and precision 0.9 in './data/eng_r0.5p0.9'

## Confidence Estimation Strategies
### Local Strategy
```bash
python confidence_estimation_local.py --dataset eng_r0.5p0.9 --embedding_file ${PATH_TO_EMBEDDING} --embedding_dim ${DIM_OF_EMBEDDING} --neg_noise_rate ${NOISE_RATE_OF_NEGATIVES} --pos_noise_rate ${NOISE_RATE_OF_POSITIVES}

```
For '--neg_noise_rate' and '--pos_noise_rate', you can set them as -1.0 to use golden noise rate, or you can set them as other values (i.e., --neg_noise_rate 0.09 --pos_noise_rate 0.14)

### Global Strategy
```bash
python confidence_estimation_global.py --dataset eng_r0.5p0.9 --embedding_file ${PATH_TO_EMBEDDING} --embedding_dim ${DIM_OF_EMBEDDING} --neg_noise_rate ${NOISE_RATE_OF_NEGATIVES} --pos_noise_rate ${NOISE_RATE_OF_POSITIVES}
```
For 'neg_noise_rate' and 'pos_noise_rate', you can set them as -1.0 to use golden noise rate, or you can set them as other values (i.e., --neg_noise_rate 0.1 --pos_noise_rate 0.13)