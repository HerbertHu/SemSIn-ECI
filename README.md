# SemSIn-ECI

Source code for ACL 2023 paper "**Semantic Structure Enhanced Event Causality Identification**"

## Requirements

It is recommended to use a virtual environment to run SemSIn-ECI.

```shell
conda create -n semsin-eci python=3.8.12
conda activate semsin-eci
```

- pytorch == 1.13.1 
- transformers
- tqdm
- scikit-learn
- lxml
- amrlib == 0.7.1
- word2number 
- unidecode
- beautifulsoup4

```shell
pip install -r requirements.txt
```

dgl install (https://www.dgl.ai/)

- dgl == 1.1.0+cu117

## Datasets

Download and unzip datasets, put them in `/data/raw_data`.

- EventStoryLine
  https://github.com/tommasoc80/EventStoryLine
- Causal-TimeBank
  https://github.com/paramitamirza/Causal-TimeBank

## Data Preprocess

### Step 1: Read documents

We provide the processed data in the directory **/data**.
The processed data file is document_raw.pkl and data_samples.pkl.

```
python read_document.py
python generate_sample.py
```

### Step 2: AMR parsing

Use the pre-trained AMR parser [parse_xfm_bart_large v0.1.0](https://github.com/bjascob/amrlib) parsing the data.
We also provide the parsed data in the directory /data/EventStoryLine/amr_data and /data/CausalTimeBank/amr_data.

```
python -m spacy download en_core_web_sm
```

```
python get_amr_data.py
```

### Step 3: Prepare data

Prepare the data that the model used.

Dataset types: ESC, ESC*, CTB.

```
/preprocess/prepare_data_esc.sh
/preprocess/prepare_data_esc_star.sh
/preprocess/prepare_data_ctb.sh
```

## Running the model

Run the model.

```
/src/run_esc.sh
/src/run_esc_star.sh
/src/run_ctb.sh
```

When the program is finished, look at the log to get the final results.

#### arguments description:

in file /src/arguments.py:

    plm_path：pretrained language model path

    save_model_name：model save name

    device_num：device number

    fold：cross-validate data number

## Citation

[Semantic Structure Enhanced Event Causality Identification](https://aclanthology.org/2023.acl-long.610) (Hu et al., ACL 2023)