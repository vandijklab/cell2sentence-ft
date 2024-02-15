# Cell2Sentence
[![Code License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Preprint License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![DOI:10.1101/2023.09.11.557287](http://img.shields.io/badge/DOI-10.1101/2023.09.11.557287-B31B1B.svg)](https://doi.org/10.1101/2023.09.11.557287)
[![Python 3.9+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)

Cell2Sentence is a novel method for adapting large language models to single-cell transcriptomics. We transform single-cell RNA sequencing data into sequences of gene names ordered by expression level, termed "cell sentences". This repository provides scripts and examples for converting cells to cell sentences, fine-tuning language models, and converting outputs back to expression values.

![Overview](https://github.com/vandijklab/cell2sentence-ft/blob/main/assets/overview.png)

## Requirements
Cell2Sentence requires Python 3.10+ and Conda. Create your python environment with `conda` (note: you need to install `conda` or `miniconda`):
```bash
conda env create -f environment.yml
conda develop .
```

Make sure to activate your conda environment with `conda activate c2s`.

## Quickstart
To get started with some sample data:
1. Download a [subset](https://drive.google.com/file/d/1PYUM59fKclw-aeN79oL5ghCkU4kn6XvN/view?usp=sharing) of `1000` cells from [1] to the `data/` directory: `python retrieve_example_data.py`.
2. Transform raw transcript counts into cell sentences: `python transform.py`.

To transform your own data, place your `.h5ad` file in the `data/` directory and run `python transform.py --data_filepath data/<your_filepath> --output_dir <your_output_dir>`. The `--output_dir` parameter lets you specify where to place the cell sentences.

The `transform.py` script creates three output directories:
- `eval/` which contains figures and evaluation metrics.
- `cell_sentences/` which contains txt files with raw cell sentences and gene vocabularies.
- `cell_sentences_hf/` which contains cell sentences and types formatted as an arrow dataset.

[1] C Domínguez Conde et al. “Cross-tissue immune cell analysis reveals tissue-specific features in humans”. In: Science 376.6594 (2022), eabl5197.

## Fine-tuning
Fine-tune a `GPT-2` model with this script:
```bash
python train.py \
    --data_dir data/cell_sentences_hf/ \
    --output_dir <your_output_dir> \
    --model_name gpt2 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --fp16 True \
    --logging_steps 32 \
    --save_steps 500
```
> By default, models are saved to the `data/model/` directory. Edit the `--data_dir` value to point to your own data directory if needed.

Switch the `model_name` to the name of any other models you'd like to fine-tune. Note that you may need to adjust the `per_device_batch_size`, `gradient_accumulation_steps`, and `gradient_checkpointing` parameters if you employ larger models. The default configuration is provided for training on a single Nvidia A5000 GPU.

## Citation
Please cite the cell2sentence paper if you use this repo.
```bibtex
@article {Levine2023.09.11.557287,
	author = {Daniel Levine and Syed Asad Rizvi and Sacha L{\'e}vy and Nazreen Pallikkavaliyaveetil MohammedSheriff and Ruiming Wu and Zihe Zhang and Antonio Fonseca and Xingyu Chen and Sina Ghadermarzi and Rahul M. Dhodapkar and David van Dijk},
	title = {Cell2Sentence: Teaching Large Language Models the Language of Biology},
	elocation-id = {2023.09.11.557287},
	year = {2023},
	doi = {10.1101/2023.09.11.557287},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/09/14/2023.09.11.557287},
	eprint = {https://www.biorxiv.org/content/early/2023/09/14/2023.09.11.557287.full.pdf},
	journal = {bioRxiv}
}
```

## Maintainers
- Sacha Lévy ([sacha.levy@yale.edu](mailto:sacha.levy@yale.edu))
- Daniel Levine ([daniel.levine@yale.edu](mailto:daniel.levine@yale.edu))
- Syed Rizvi ([syed.rizvi@yale.edu](mailto:syed.rizvi@yale.edu))
