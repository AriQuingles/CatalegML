---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:10000
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: CARGOL CORT.3,5 X34(HAQ03-02-3534), COLZE NO TUMORAL
  sentences:
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PLACA HÚMER
    PROXIMAL ESQUERRA 3,5 PERI-LOC 11FORATS 191MM ESTÈRIL, ACER
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PLACA HÚMER
    PROXIMAL ACUMED PHP EXTRALLARGA DRETA, TITANI. REFERÈNCIA PL-PHXGR
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PLACA RADI
    2,5, 11 FORATS, ESTRETA, PALMAR, ESQUERRE, LLARGA XL, TITANI. AGOMED. REF; 4002606
- source_sentence: CARGOL ALPS CORTICAL 3,5X12 (815037012), OSTEOSINTESI
  sentences:
  - 'PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL
    CORTICAL  Ø2,7X10MM,AUTORROSCANT,CAP STARDRIVE T8,ACER INOXIDABLE,SISTEMA LCP
    REF: 202.870'
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL CORT
    PLFE DIST 4-5MM V/L214.885TS
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL DE
    BLOQUEIG PER A PLACA DE COMPRESSIÓ I BLOQUEIG LCP,Ø3,5X14MM DE LONGITUD,AUTORROSCANT,CAP
    HEXAGONAL,ACER INOXIDABLE.SISTEMA  LCP REF:213.014
- source_sentence: CAP TULIPA MULTIAX.RELINE MAS(16171111), COLUMNA FIX VERT
  sentences:
  - 'PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL
    DE BLOQUEIG Ø3,5X85MM DE LONGITUD,AUTORROSCANT,CAP CÒNIC  STARDRIVE T15, ACER
    INOXIDABLE, SISTEMA LCP REF: 212.129'
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL CORTICAL
    DE 2,4 MM X 9 MM DE LLARG ,  ESTÈRIL,D'ACER.SISTEMA EVOS PETIT FRAGMENT, REF:72402409
  - PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, CAIXA INTERSOMÀTICA TLIF
    MONOPORTAL LORDÒTICA 11X10X27MM PEEK REFORÇAT AMB FIBRA DE CARBONO CONCORDE REF:187827510
- source_sentence: CARGOL D/ROSCA PHOENIX 5X75(14405075), TURMELL
  sentences:
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL DE
    ESPONJOSA ROSCA PARCIAL Ø 4,0X80MM TITANI, SISTEMA PETITS FRAGMENTS ALPS BIOMET
    REF. 815540080
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PLACA DE
    COMPRESSIÓ BLOQUEJADA PER A HÚMER DISTAL MEDIAL 3,5,ESQUERRA, 58MM DE LONGITUD,3
    FORATS,ACER INOXIDABLE.SISTEMA LCP 2,7/3,5 REF:241.283
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PLACA DE
    COMPRESSIÓ I BLOQUEIG D'ANGLE VARIABLE PER A HÚMER DISTAL MEDIAL, ESQUERRA, CURTA,
    69MM DE LONGITUD, 1 FORAT, ACER INOXIDABLE, SISTEMAVA-LCP 2,7/3,5 REF:02.117.501S
- source_sentence: CARGOL HQ COMPR.NEXIS PECA 4X46(PS050046), TURMELL
  sentences:
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PL FRAG
    ESP CÚB I RAD DIS AR-8916SPN-ST
  - 'PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE MINI FRAGAMENTS, PLACA BLOQUEIG
    PER A COLL METACARPIÀ, 1.5MM DE GRUIX, 29 MM DE LONGITUD,DORSAL, 8 FORATS, TITANI
    PUR , SISTEMA AV HAND 1.5, REF: 04.130.268'
  - PRÒTESIS OSTEOSÍNTESI GENERAL, CARGOLS CANULATS, CARGOL CANULAT AUTOPERFORANT,AUTORROSCANT,
    Ø8.0MMX 110MM DE LONGITUD, ROSCA PARCIAL DE 16MM DE LONGITUD, CAP DE BAIX PERFIL,ACABAMENT
    DEL CARGOL EN TRES PUNTES,CODI DE COLORS,SISTEMA TIMAX ,ALIATGE DE TITANI TI6AI4V,REF:110007926
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 512-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 512 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: DistilBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Dense({'in_features': 768, 'out_features': 512, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'CARGOL HQ COMPR.NEXIS PECA 4X46(PS050046), TURMELL',
    'PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PL FRAG ESP CÚB I RAD DIS AR-8916SPN-ST',
    'PRÒTESIS OSTEOSÍNTESI GENERAL, CARGOLS CANULATS, CARGOL CANULAT AUTOPERFORANT,AUTORROSCANT, Ø8.0MMX 110MM DE LONGITUD, ROSCA PARCIAL DE 16MM DE LONGITUD, CAP DE BAIX PERFIL,ACABAMENT DEL CARGOL EN TRES PUNTES,CODI DE COLORS,SISTEMA TIMAX ,ALIATGE DE TITANI TI6AI4V,REF:110007926',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 512]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 10,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                               |
  | details | <ul><li>min: 22 tokens</li><li>mean: 33.49 tokens</li><li>max: 46 tokens</li></ul> | <ul><li>min: 48 tokens</li><li>mean: 100.71 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                | sentence_1                                                                                                                                                                                                                                         |
  |:--------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>CARGOL RELINE MAS POLIAX 6,5X50 (16016550), COLUMNA FIX VERT</code> | <code>PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, CARGOL POLIAXIAL PER A SISTEMA COLUMNA TORACO LUMBAR POSTERIOR SILVER,Ø7.5MMX45MM DE LONGITUD,ESTÈRIL, TITANI  I COBRIMENT AMB NANOPARTÍCULES D'IÓ DE PLATA, REF:AB-NSPA7545</code> |
  | <code>CARGOL SOLA FRS 3 X 14(110018450), PEU</code>                       | <code>PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PLACA PREC  PE POST Ø3,5 A-4854.01</code>                                                                                                                              |
  | <code>CARGOL ALPS CORTICAL 3,5X12 (815037012), OSTEOSINTESI</code>        | <code>PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL ESTÀNDARD DIÀMETRE 3,5MM X 45MM LONGITUD, HEXAGONAL, SISTEMA NORMED, TITANI</code>                                                                              |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4    | 500  | 0.4005        |
| 0.8    | 1000 | 0.298         |
| 0.4    | 500  | 0.3752        |
| 0.8    | 1000 | 0.2609        |
| 0.4    | 500  | 0.3945        |
| 0.8    | 1000 | 0.2671        |
| 0.4    | 500  | 0.4304        |
| 0.8    | 1000 | 0.2681        |
| 0.4    | 500  | 0.3876        |
| 0.8    | 1000 | 0.289         |
| 0.4    | 500  | 0.382         |
| 0.8    | 1000 | 0.2737        |
| 0.4    | 500  | 0.3875        |
| 0.8    | 1000 | 0.305         |
| 0.4    | 500  | 0.4055        |
| 0.8    | 1000 | 0.2407        |
| 0.4    | 500  | 0.3869        |
| 0.8    | 1000 | 0.2424        |
| 0.4    | 500  | 0.4021        |
| 0.8    | 1000 | 0.2233        |
| 0.4    | 500  | 0.4249        |
| 0.8    | 1000 | 0.2782        |
| 0.4    | 500  | 0.3874        |
| 0.8    | 1000 | 0.2433        |
| 0.4    | 500  | 0.397         |
| 0.8    | 1000 | 0.2632        |
| 0.4    | 500  | 0.4065        |
| 0.8    | 1000 | 0.2267        |
| 0.4    | 500  | 0.368         |
| 0.8    | 1000 | 0.256         |
| 0.4    | 500  | 0.3573        |
| 0.8    | 1000 | 0.2475        |
| 0.4    | 500  | 0.3838        |
| 0.8    | 1000 | 0.3046        |
| 0.4    | 500  | 0.3711        |
| 0.8    | 1000 | 0.2466        |
| 0.4153 | 500  | 0.3826        |
| 0.8306 | 1000 | 0.235         |


### Framework Versions
- Python: 3.12.7
- Sentence Transformers: 3.4.1
- Transformers: 4.49.0
- PyTorch: 2.6.0+cpu
- Accelerate: 1.5.2
- Datasets: 3.4.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->