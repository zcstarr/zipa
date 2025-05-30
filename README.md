# ZIPA: A family of efficient speech models for multilingual phone recognition - ACL 2025

[Paper](https://arxiv.org/abs/2505.23170)

## Inference

### Pretrained models

## Data

All data are distributed in the scalable `shar` format, similar to `webdataset` format but with indexes. It can be easily loaded with `lhotse` library. Audio files are downsampled to 16000Hz and stored in the `flac` format to save space. 

 - [All processed data](https://huggingface.co/collections/anyspeech/ipapack-raw-673c2d345deec72e82e28a3b) (~1.8TB)
 - [Training data only](https://huggingface.co/collections/anyspeech/ipa-pack-train-6838a6804a3a71a91794a801) (~1.5TB)
 - [Pseudolabeled data](https://huggingface.co/collections/anyspeech/ipa-pack-train-pseudolabel-6838a6adc3ccad443cfb63b0) (~1TB)

After downloading all data, place all `tar` and `json` files within the same folder. 

```
data-shar
├── cuts.000000.jsonl.gz
├── recording.000000.tar
```
Then you can construct a data loader with `lhotse`. Please refer to the `lhotse` documentation and [their shar tutorial](https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/04-lhotse-shar.ipynb) for further details. 
```
cuts_full = CutSet.from_shar(
    fields={
        "cuts": ["data-shar/cuts.000000.jsonl.gz"],
        "recording": ["data-shar/recording.000000.tar"],
    }
)
```

## Training

