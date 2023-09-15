# Cell2Sentence

Cell2Sentence is a novel method for adapting large language models to single-cell transcriptomics. We transform single-cell RNA sequencing data into sequences of gene names ordered by expression level, termed "cell sentences". This repository provides scripts and examples for converting cells to cell sentences, fine-tuning language models, and converting outputs back to expression values.

![Overview](https://github.com/vandijklab/cell2sentence-ft/blob/main/Overview.png)



## Quickstart

1. Download the data subset used in the example here:
    - Domínguez Conde, C., et al. [Cross-tissue immune cell analysis reveals tissue-specific features in humans](https://drive.google.com/file/d/1PYUM59fKclw-aeN79oL5ghCkU4kn6XvN/view?usp=sharing)
    - Place the data in the root of the data repository

2. Run preprocessing: `python preprocessing.py`

3. Run `python create_cell_sentence_arrow_dataset.py`

## Fine-tuning

Once the arrow dataset is created, you can create a json file containing the paths of the train and validation datasets, for example:

```
{
    'train': <PATH_TO_TRAINING_DATASET>,
    'val': <PATH_TO_VALIDATION_DATASET>
}
```

Then to train Hugging Face's GPT-2 small model using our fine-tuning script, run:

```
python finetune.py \
    --output_dir <OUTPUT_DIRECTORY> \
    --datasets_paths <PATH_TO_DATASET_PATHS_JSON> \
    --model_name gpt2 \
    --num_train_epochs 100 \
    --fp16 True \
    --dataloader_num_workers 2 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --logging_steps 100 \
    --eval_steps 100 \
    --eval_dataset_size 1000 \
    --save_steps 500
```
