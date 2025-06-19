# ZIPA: A family of efficient speech models for multilingual phone recognition - ACL 2025

[Paper](https://arxiv.org/abs/2505.23170)

## Environment

A pre-built apptainer container can be found [here](https://huggingface.co/datasets/anyspeech/ipapack_plus_meta/blob/main/pytorch2.4.0-cuda12.4-icefall-container.sif).

## Inference

### Pretrained models

## Data

The tokenizer can be found [here](https://huggingface.co/datasets/anyspeech/ipapack_plus_meta). You'll need the `sentencepiece` package to load it. [This](https://huggingface.co/datasets/anyspeech/ipapack_plus_meta/blob/main/ipa_simplified/unigram_127.vocab) is the list of selected IPA symbols. 

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
Training a Zipformer-Large CRCTC model

```
python zipformer_crctc/train.py --world-size 2 --num-epochs 2 --start-epoch 1 --start-batch 500000 --use-fp16 1 --exp-dir /lustre07/scratch/lingjzhu/zipformer_exp/zipformer_large_crctc_0.5_scale --causal 0 --full-libri True --max-duration 120 --use-transducer False --use-ctc True  --use-cr-ctc True --base-lr 0.015  --enable-spec-aug False --seed 2333 --wandb False --num-encoder-layers 4,3,4,5,4,4 --feedforward-dim 768,768,1536,2048,1536,768 --encoder-dim 512,512,768,1024,768,512 --encoder-unmasked-dim 192,192,256,320,256,192 --decoder-dim 1024 --joiner-dim 1024 --query-head-dim 64 --value-head-dim 48 --num-heads 6,6,6,8,6,6 --num-buckets 8 --num-workers 4 --unsup-cr-ctc-loss-scale 0.5 --use-unsup-cr-ctc True
```

Remove diacritics
```
python zipformer_crctc/train.py --world-size 2 --num-epochs 2 --start-epoch 1 --start-batch 500000 --use-fp16 1 --exp-dir /lustre07/scratch/lingjzhu/zipformer_exp/zipformer_large_crctc_0.5_scale_no_diacritics --causal 0 --full-libri True --max-duration 120 --use-transducer False --use-ctc True  --use-cr-ctc True --base-lr 0.015  --enable-spec-aug False --seed 2333 --wandb False --num-encoder-layers 4,3,4,5,4,4 --feedforward-dim 768,768,1536,2048,1536,768 --encoder-dim 512,512,768,1024,768,512 --encoder-unmasked-dim 192,192,256,320,256,192 --decoder-dim 1024 --joiner-dim 1024 --query-head-dim 64 --value-head-dim 48 --num-heads 6,6,6,8,6,6 --num-buckets 8 --num-workers 4 --unsup-cr-ctc-loss-scale 0.5 --use-unsup-cr-ctc True --remove-diacritics True
```
