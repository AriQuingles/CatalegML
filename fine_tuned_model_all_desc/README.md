---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:10000
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: CARGOL EVOS OST SCR F-T 4,7X22(72424722), OSTEOSINTESI
  sentences:
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL CORTICAL
    AUTORROSCANT Ø4.5X 60MM DE LONGITUD,ROSCA TOTAL,1,75 PAS DE ROSCA,CAP HEXAGONAL,ACER
    INOXIDABLE,SISTEMA SURGIVAL REF:3308-060
  - PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, CARGOL MULTIAXIAL LEGACY
    6,35 ACER Ø 7,5X40 REF.76647540
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PLACA HÚMER
    PROXIMAL DRETA,120MM DE LONGITUD,5FORATS A CAP, 6FORATS TIJA,ANATÒMICA,BAIX PERFIL,DISPOSA
    DE GANXET AL CAP PER SUTURAR,ALIATGE DE TITANI, SISTEMA PANTERA REF:TO-PHP-R120
- source_sentence: CARGOL EVOS CTX SCR S-T 2,7 X13(72402713), OSTEOSINTESI
  sentences:
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL DE
    BLOQUEIG Ø 4,0 X 36MM AUTORROSCANT CAP T15(Ø15MM) PER A PLACA HUMERAL PROXIMAL
    DE BLOQUEIG AXSOS ACER INOXIDABLE
  - PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, CAIXA INTERSOMÀTICA PER
    A SISTEMA COLUMNA TORACO LUMBAR PEZO-P, ANGLE 5°, AMPLADA 11MM, ALÇADA 9MM, LONGITUD
    29MM, PEEK, REF:CS 3315-09
  - PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, CARGOL MONO-AXIAL PER
    SUBJECCIÓ DE BARRA 6.5 MM X 40 MM DE TITANI BLACKSTONE REF:55-4640
- source_sentence: CARGOL ALPS CORT.BLOQ.3,5X44 (816135044), OSTEOSINTESI
  sentences:
  - PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, BARRA CORBADA PER A SISTEMA
    DE COLUMNA TORACO LUMBAR SACROILIACA CREO MIS Ø5,5MMX120MM,REF:1134.7120
  - PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, GANXO PEDICULAR LÀMINA
    ESTRETA 6,5 PER SISTEMA DE FIXACIÓ COLUMNA TORACO LUMBAR EXPEDIUM, ACER INOXIDABLE,
    REF:188152036
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL ESTÀNDARD
    DIÀMETRE 3,5MM X 22MM LONGITUD, HEXAGONAL, SISTEMA NORMED, TITANI
- source_sentence: CARGOL PED.POLI.MOSS100 7X45(107-018-7045), COLUMNA FIX VERT
  sentences:
  - PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, CARGOL MULTI-AXIAL ESTÀNDARD
    PER SUBJECCIÓ DE BARRA 6.5 MM X 35 MM DE TITANI BLACKSTONE REF 56-3635
  - PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, CARGOL MODULAR AMB DOBLE
    PAS DE ROSCA PER A SISTEMA DE FIXACIÓ VERTEBRAL MODEL SFS FIREBIRD, 5,5MM DIÀMETRE
    X 55MM LLARG, TITANI, REF:44-5555
  - 'PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE MINI FRAGAMENTS, PLACA DE
    BLOQUEIG EN T  PER A CIRURGIA DE MÀ,AMPLA,  8 FORATS, 1,5MM GRUIX, TITANI. SISTEMA
    VARIAX MÀ 2,3 REF: 5715361'
- source_sentence: TUERCA BLOQ.PERLA (TLF-SC 02 00 S), COLUMNA FIX VERT
  sentences:
  - 'PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, COLUMNA TORACO-LUMBAR-SACROILÍACA:
    CAIXES-INTERSOMÀTICS ABORDATGE ANTERIOR ALIF SENSE CARGOL PER SISTEMA AVILA -
    O,REF:369210806'
  - PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE FORMAT GENERAL, CARGOL DE
    BLOQUEIG DE 2,7 MM X 18 MM DE LLARG , NO ESTÈRIL,D'ACER.SISTEMA EVOS PETIT FRAGMENT,
    REF:72412718N
  - PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, CARGOL PEDICULAR PER
    A ARTRODESI POLIAXIAL 5,0X40MM TITANI EXPEDIUM SI REF. 179712540
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
    'TUERCA BLOQ.PERLA (TLF-SC 02 00 S), COLUMNA FIX VERT',
    'PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, COLUMNA TORACO-LUMBAR-SACROILÍACA: CAIXES-INTERSOMÀTICS ABORDATGE ANTERIOR ALIF SENSE CARGOL PER SISTEMA AVILA - O,REF:369210806',
    'PRÒTESI DE COLUMNA, COLUMNA TORACO-LUMBAR-SACROIL·LÍACA, CARGOL PEDICULAR PER A ARTRODESI POLIAXIAL 5,0X40MM TITANI EXPEDIUM SI REF. 179712540',
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
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 24 tokens</li><li>mean: 34.96 tokens</li><li>max: 45 tokens</li></ul> | <ul><li>min: 34 tokens</li><li>mean: 96.1 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                               | sentence_1                                                                                                                                                                                                                            |
  |:-------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>CARGOL ALPS CORT.BLOQ.3,5X38 (816135038), OSTEOSINTESI</code>      | <code>PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, CARGOL CORTICAL Ø3.5X40MM DE LONGITUD,AUTORROSCANT,CAP HEXAGONAL T20,ESTÈRIL,ACER INOXIDABLE 316L,SISTEMA PERI-LOC PETITS FRAGMENTS REF: 7180-4040</code> |
  | <code>CARGOL ALPS CORT.BLOQ.3,5X44 (816135044), OSTEOSINTESI</code>      | <code>PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE PETITS FRAGMENTS, PLACA RECONSTRUCC P FRAG TO-CLP-LBP-226</code>                                                                                                            |
  | <code>CARGOL POLIAX.SILVER 6,5X45 (AB-NSPA6545), COLUMNA FIX VERT</code> | <code>PRÒTESIS OSTEOSÍNTESI GENERAL, PLAQUES I CARGOLS DE FORMAT GENERAL, CARGOL ESPONJOSA ROSCA CURTA Ø 6,0 X 55MM /LR16MM SISTEMA 5.0  PER A PLAQUES  D¿ESTABILITAT ANGULAR AXSOS 3,TITANI , REF: 608255</code>                     |
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
| 0.4    | 500  | 0.9084        |
| 0.8    | 1000 | 0.8848        |
| 0.4    | 500  | 0.8968        |
| 0.8    | 1000 | 0.8994        |
| 0.4    | 500  | 0.9071        |
| 0.8    | 1000 | 0.8747        |
| 0.4    | 500  | 0.9151        |
| 0.8    | 1000 | 0.9188        |
| 0.4    | 500  | 0.8696        |
| 0.8    | 1000 | 0.8717        |
| 0.4    | 500  | 0.878         |
| 0.8    | 1000 | 0.8615        |
| 0.4    | 500  | 0.8741        |
| 0.8    | 1000 | 0.8547        |
| 0.4    | 500  | 0.8722        |
| 0.8    | 1000 | 0.8448        |
| 0.4    | 500  | 0.8485        |
| 0.8    | 1000 | 0.8468        |
| 0.4    | 500  | 0.8372        |
| 0.8    | 1000 | 0.8162        |
| 0.4    | 500  | 0.8414        |
| 0.8    | 1000 | 0.7922        |
| 0.4    | 500  | 0.7989        |
| 0.8    | 1000 | 0.7679        |
| 0.4    | 500  | 0.8105        |
| 0.8    | 1000 | 0.7533        |
| 0.4    | 500  | 0.7975        |
| 0.8    | 1000 | 0.7367        |
| 0.4456 | 500  | 0.7909        |
| 0.8913 | 1000 | 0.7369        |


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