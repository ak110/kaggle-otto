# kaggle otto

Feature engineeringの要らない(?)テーブルデータということでLate submissionしてみているコード。

<https://www.kaggle.com/c/otto-group-product-classification-challenge>

## data

```bash
kaggle competitions download -c otto-group-product-classification-challenge -p data
unzip data/otto-group-product-classification-challenge.zip -d data/
```

## submit

```bash
kaggle c submit --file=models/lv2_lgb/submission.csv --messag="CV 0.xxx" otto-group-product-classification-challenge
```

## submissions

```bash
kaggle c submissions otto-group-product-classification-challenge
```

## memo

|    # |    CV |  Public | Private |
| ---: | ----: | ------: | ------: |
|    1 | 0.422 | 0.41108 | 0.41509 |
