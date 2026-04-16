## Setup

### Load required modules

import os
import sys
import tempfile
import shutil
import gc
from collections import Counter
import json

from datetime import datetime
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, precision_recall_fscore_support, balanced_accuracy_score

from datasets import Dataset, DatasetDict

import torch
from torch.nn import CrossEntropyLoss

from transformers import (
  set_seed,
  AutoTokenizer,
  AutoModelForSequenceClassification,
  TrainingArguments,
  Trainer,
  DataCollatorWithPadding,
  PrinterCallback,
  EarlyStoppingCallback,
)
from transformers import EvalPrediction
from transformers.trainer_utils import PredictionOutput

import warnings

from typing import Union, List, Dict, Callable

### Helper functions

#### Reporting

# function that prints a time stamp as string
ts = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log = lambda x: print(f'[{ts()}] {x}')

def compute_label_distribution(df, label_col, id2label: Union[None,dict], as_string=False):
  counts = dict(df[label_col].value_counts(normalize=False))
  props = dict(df[label_col].value_counts(normalize=True))
  if id2label:
    counts = {id2label[k]: v for k, v in counts.items()}
    props = {id2label[k]: v for k, v in props.items()}
  out = {l: (p, c) for (l, c), p in zip(counts.items(), props.values())}
  if as_string:
    out = ', '.join([f"'{l}' = {p:.2f} ({c})" for l, (p, c) in out.items()])
  return out

#### Data splitting

def downsample(df, minority_class, frac, seed, label_col='label'):
  pos_share = np.mean(df[label_col] == minority_class)
  if pos_share >= 0.5:
    raise ValueError(f"the label '{minority_class}' is not the minority class in the training data")
  if pos_share > frac:
    raise ValueError(f"the label '{minority_class}' already makes up {pos_share*100}% of the training data. Increase --downsample_pos_fraction !")

  if len(df[label_col].unique()) > 2:
    print('WARNING: setting --downsample_pos_fraction only supported for binary classification. Applying downsampling to all non-minority classes.')
    downsampling_strategy = 'not minority'
  else:
    downsampling_strategy = frac

  downsampler = RandomUnderSampler(sampling_strategy=downsampling_strategy, random_state=seed)
  df, _ = downsampler.fit_resample(df, df[label_col].astype(str))

  return df

def split_data(
    df,
    dev_size: float,
    test_size: float,
    sampling_strategy: str,
    stratify_by_cols: Union[str, None],
    downsample_train_data: bool,
    minority_fraction: Union[float, None],
    minority_class: Union[str, None],
    seed: int,
):
  if sampling_strategy not in ['random', 'stratified']:
    raise ValueError(f"invalid value '{sampling_strategy}' passed to `--sampling_strategy`")
  if 'stratified' in sampling_strategy and stratify_by_cols:
    # combine a strata indicator from unique values of stratify_by_cols
    df['strata_'] = df[stratify_by_cols].apply(lambda x: '_'.join([str(v) for v in x]), axis=1)
  
  # split such that set sizes match test_size and dev_size props
  n = len(df)
  n_test = int(n*test_size)
  tmp, test_idxs = train_test_split(range(n), test_size=n_test, random_state=seed, stratify=df.strata_ if sampling_strategy == 'stratified' else None)
  n_dev = int(n*dev_size)
  train_idxs, dev_idxs = train_test_split(tmp, test_size=n_dev, random_state=seed, stratify=df.strata_.iloc[tmp] if sampling_strategy == 'stratified' else None)
  del tmp
  train_df = df.iloc[train_idxs]
  dev_df = df.iloc[dev_idxs]
  test_df = df.iloc[test_idxs]
  # # DPERERCATED
  # tmp_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df.strata_ if sampling_strategy == 'stratified' else None)
  # train_df, dev_df = train_test_split(tmp_df, test_size=dev_size, random_state=seed, stratify=tmp_df.strata_ if sampling_strategy == 'stratified' else None)
  # del tmp_df

  if downsample_train_data:
    train_df = downsample(train_df, minority_class, minority_fraction, seed)

  return train_df, dev_df, test_df

#### Training utils

class TrainerWithClassWeights(Trainer):
  '''
  Trainer class that allows accounts for label calss weights in loss function

  Args:
      device (Union[torch.device, str]): device to put the model on
      class_weights (Union[List, Dict]): class weights to use in loss function.
                                          If a list is passed, it must be of length `num_labels` and the order of the weights must match the order of the labels in the model config.
                                          If a dict is passed, it must have the labels as keys and the weights as values.
  '''
  def __init__(self, device: Union[torch.device, str], class_weights: Union[List, Dict], **kwargs):
    super().__init__(**kwargs)
    self.device = device if isinstance(device, torch.device) else torch.device(device)
    self.model.to(self.device);
    if len(class_weights) != self.model.config.num_labels:
      raise ValueError(f"length of `class_weights` must be {self.model.config.num_labels}")
    if isinstance(class_weights, dict):
      if set(class_weights.keys()) != set(self.model.config.id2label.keys()):
        raise ValueError(f"keys of `class_weights` mismatch label classes {list(self.model.config.id2label.keys())}")
      class_weights = [v for k, v in sorted(class_weights.items(), key=lambda item: item[1])]
    self.class_weights = torch.tensor(class_weights, dtype=self.model.dtype)

  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get('labels')
    # forward pass
    outputs = model(**inputs)
    logits = outputs.get('logits')
    # compute custom loss
    class_weights = self.class_weights.to(self.device)
    loss_fct = CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

def compute_class_weights(data: Dataset, label_name='label'):
  labs = data[label_name].tolist() if isinstance(data[label_name], torch.Tensor) else data[label_name]
  cnts = dict(Counter(labs))
  weights = len(data)/np.array(list(cnts.values()))
  weights = weights/sum(weights)
  class_weights = {l: w for l, w in zip(cnts.keys(), weights)}
  return class_weights

def clean_memory(device):
  if device == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
  elif device == 'mps':
    torch.mps.empty_cache()
    gc.collect()
  else:
    pass

#### Evaluation

def classification_metrics_binary(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    label2id: Dict,
    pos_label: Union[str, int] = 1
):

  if isinstance(pos_label, str):
    pos_label = label2id[pos_label]

  with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    res = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0.0)
    metrics = {}
    metrics['f1'] = res[2]
    metrics['precision'] = res[0]
    metrics['recall'] = res[1]
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['prevalence'] = np.mean(y_true == pos_label)

  return metrics

def classification_metrics_multiclass(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    label2id: dict,
    **kwargs
  ):
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    res = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=list(label2id.values()),
        target_names=list(label2id.keys()),
        output_dict=True,
        zero_division=0.0
      )
    for l in label2id.keys():
      res[l]['prevalence'] = res[l]['support']/len(y_true)

    metrics = {}

    # aggregate metrics
    metrics['f1_macro'] = res['macro avg']['f1-score']
    metrics['f1_micro'] = res['accuracy']
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

  # class-wise metrics
  metrics.update(
    {f"{l}_{m.replace('-score', '')}": res[l][m] if l in res else np.nan
      for l in label2id.keys()
      for m in ['f1-score', 'precision', 'recall', 'prevalence']
    }
  )

  return metrics

def prepare_predictions_singlelabel(p: Union[EvalPrediction, PredictionOutput], **kwargs):
  preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
  preds = preds.argmax(-1)
  return(preds)

def compute_metrics_function(
	p: Union[EvalPrediction, PredictionOutput],
	eval_fun: Callable,
	label2id: Dict,
	**kwargs
):
	preds = prepare_predictions_singlelabel(p)
	return eval_fun(p.label_ids, preds, label2id=label2id, **kwargs)

def bootstrap_metrics_function(
	p: Union[EvalPrediction, PredictionOutput],
	eval_fun: Callable,
	n_bootstraps: int = 50,
	seed: int = 1234,
	**kwargs
):
  preds = prepare_predictions_singlelabel(p)
  rng = np.random.default_rng(seed)
  n_ = p.label_ids.shape[0]

  bs = list()
  for i in range(n_bootstraps):
    idxs = rng.choice(np.arange(n_), size=n_, replace=True, p=None)
    bs.append(eval_fun(p.label_ids[idxs], preds[idxs], **kwargs))
  metrics = {m: [] for m in bs[0].keys()}
  for b in bs:
    for m, s in b.items():
      metrics[m].append(s)
  return metrics

def evaluate(
    trainer: Trainer,
    dataset: Dataset,
    id_col: Union[None, str],
    eval_fun: Callable,
    eval_by: Union[None, str],
    eval_by_values: Union[None, List[str]],
    label2id: Dict,
    id2label: Union[None, Dict] = None,
    **kwargs
):
  # inference on test set samples
  predictions = trainer.predict(dataset)
  
  # get labels and predictions
  labels = predictions.label_ids
  preds = prepare_predictions_singlelabel(predictions)
  if id2label is None:
    id2label = {i: l for l, i in label2id.items()}
  labels = [id2label[int(l)] for l in labels]
  preds = [id2label[int(l)] for l in preds]
  
  # bootstrap eval scores
  bsm = bootstrap_metrics_function(
    p=predictions,
    eval_fun=eval_fun,
    label2id=label2id,
    **kwargs
  )

  # compute statistics for bootstrapped eval scores
  bsm_sum = {}
  for k, v in bsm.items():
    bsm_sum[k] = [np.mean(v)]
    bsm_sum[k] += np.quantile(v, [.025, .975]).tolist()

  # compute bootstrapped eval metrics by group (if needed)
  groups = None
  if eval_by:
    groups = dataset[eval_by]
    grouped_bsm = {}
    grouped_eval = {}
    for grp in np.unique(groups):
      if eval_by_values and grp not in eval_by_values:
        continue
      idxs = np.array([i for i, g in enumerate(groups) if g == grp])
      tmp = EvalPrediction(
        predictions=predictions.predictions[idxs],
        label_ids=predictions.label_ids[idxs]
      )
      grouped_bsm[grp] = bootstrap_metrics_function(
        tmp,
        eval_fun=eval_fun,
        label2id=label2id,
        **kwargs
      )
      grouped_eval[grp] = {}
      for k, v in grouped_bsm[grp].items():
        grouped_eval[grp][k] = [np.mean(v)]
        grouped_eval[grp][k] += np.quantile(v, [.025, .975]).tolist()
      grouped_eval[grp]['size'] = len(idxs)
    grouped_eval['overall'] = bsm_sum

  ids = dataset[id_col] if id_col is not None else None 
  out = {
    'metrics': {m.replace('test_', ''): v for m, v in predictions.metrics.items()},
    'bootstrapped': {
      'overall': bsm,
      'grouped': grouped_bsm if eval_by else None,
      'summarized': grouped_eval if eval_by else {'overall': bsm_sum}
    },
    'labels': {
      'label': labels if isinstance(labels, list) else labels.cpu().tolist() if isinstance(labels, torch.Tensor) else labels.tolist() if isinstance(labels, np.ndarray) else None,
      'pred': preds,
      'group': groups if isinstance(groups, list) else groups.cpu().tolist() if isinstance(groups, torch.Tensor) else groups.tolist() if isinstance(groups, np.ndarray) else None,
      'id': ids if isinstance(ids, list) else groups.cpu().tolist() if isinstance(groups, torch.Tensor) else groups.tolist() if isinstance(groups, np.ndarray) else None
    }
  }

  return out

def main(args):

  # ----------------------------------------------------------------
  # - Prepare data -------------------------------------------------
  # ----------------------------------------------------------------

  # determine file format
  file_extension = args.data_file.split('.')[-1]

  match file_extension:
    case 'csv':
      sep = ','
    case 'tsv':
      sep = '\t'
    case _:
      raise ValueError(f'can\'t handle input data file format ".{file_extension}"')
  
  # for reproducibility
  set_seed(args.seed)

  # read file
  df = pd.read_csv(args.data_file, sep=sep)

  # check if any in label_col are nan
  n_na = sum(df[args.label_col].isna())
  if n_na > 0:
    log(f"Removing {n_na} rows with NA values on label column '{args.label_col}'")
    df = df[~df[args.label_col].isna()]

  # keep only required columns
  cols = [args.text_col, args.label_col]
  if args.id_col:
      cols.append(args.id_col)
  if args.filter_by_col:
      cols.append(args.filter_by_col)
  if args.stratify_by_cols:
      if args.stratify_by_cols is None:
          raise ValueError('if `--stratify_by_cols` is set, `--stratify_by_cols` must be set as well')
      if isinstance(args.stratify_by_cols, str):
          args.stratify_by_cols = args.stratify_by_cols.strip().split(',')
      cols += args.stratify_by_cols
  if args.eval_by:
      cols.append(args.eval_by)
  cols = list(set(cols))
  
  # check if all required columns are present
  missing = [col for col in cols if col not in df.columns]
  if len(missing) > 0:
      raise ValueError(f"the following columns are missing from the data file: {', '.join(missing)}")
  
  # select required columns
  df = df[cols]
  
  # apply label value filtering if necessary
  if args.label_values:
      if isinstance(args.label_values, str):
          args.label_values = [v.strip() for v in args.label_values.strip().split(',')]
      log(f"Filtering label column '{args.label_col}' by values [{', '.join(args.label_values)}]")
      vals = df[args.label_col].unique()
      if isinstance(args.label_values, list):
        if not all([v in vals for v in args.label_values]):
            raise ValueError(f"some of the values passed to `--label_values` are not present in column '{args.filter_by_col}': {', '.join([v for v in args.label_values if v not in vals])}")
        df = df[df[args.label_col].isin(args.label_values)]

  # apply filtering if necessary
  if args.filter_by_col:
      if args.filter_value is None:
          raise ValueError('if `--filter_by_col` is set, `--filter_value` must be set as well')
      if isinstance(args.filter_value, str):
          args.filter_value = [v.strip() for v in args.filter_value.strip().split(',')]
      log(f"Filtering data by '{args.filter_by_col}' values [{', '.join(args.filter_value)}]")
      vals = df[args.filter_by_col].unique()
      if isinstance(args.filter_value, list):
        if not all([v in vals for v in args.filter_value]):
            raise ValueError(f"some of the values passed to `--filter_value` are not present in column '{args.filter_by_col}': {', '.join([v for v in args.filter_value if v not in vals])}")
        df = df[df[args.filter_by_col].isin(args.filter_value)]
      else:
        if args.filter_value not in vals:
          raise ValueError(f"the value passed to `--filter_value` is not present in column '{args.filter_by_col}'")
        df = df[df[args.filter_by_col] == args.filter_value]

  # check if any in text_col are nan
  n_na = sum(df[args.text_col].isna())
  if n_na > 0:
    log(f"removing {n_na} rows with NA values on text column '{args.text_col}'")
    df = df[~df[args.text_col].isna()]

  # convert boolean label column to string if needed
  if any(df[args.label_col].dtype.name == t for t in ['bool', 'object']):
    log(f"Converting values in label column '{args.label_col}' to string values")
    df[args.label_col] = df[args.label_col].astype(str)
  
  if args.text_col != 'text':
      df.rename(columns={args.text_col: 'text'}, inplace=True)
  if args.label_col != 'label':
      df.rename(columns={args.label_col: 'label'}, inplace=True)
  
  # apply preprocessing if necessary
  if args.text_preprocessing == 'twitter':
      log('applying twitter preprocessing')
      def preprocess(text):
          new_text = []
          for t in text.split(" "):
              t = '@user' if t.startswith('@') and len(t) > 1 else t
              t = 'http' if t.startswith('http') else t
              new_text.append(t)
          return " ".join(new_text)
      df['text'] = df['text'].apply(preprocess)
  elif args.text_preprocessing is not None:
      raise ValueError(f"value '{args.text_preprocessing}' passed to --text_preprocessing not supported")
  
  # create label-2-id mapping
  id2label = dict(enumerate(df.label.unique()))
  label2id = {v: k for k, v in id2label.items()}
  
  # encode label indicator
  df.loc[:,'label'] = df['label'].map(label2id)
  
  train_df, dev_df, test_df = split_data(
    df,
    dev_size=args.dev_size,
    test_size=args.test_size,
    sampling_strategy=args.sampling_strategy,
    stratify_by_cols=args.stratify_by_cols,
    downsample_train_data=args.downsample_train_data,
    minority_fraction=args.downsample_minority_ratio if args.downsample_train_data else None,
    minority_class=label2id[args.minority_label] if args.downsample_train_data else None,
    seed=args.seed
  )
  
  if args.test_mode:
    train_df = train_df.sample(frac=0.2, random_state=args.seed)
    dev_df = dev_df.sample(frac=0.2, random_state=args.seed)
    test_df = test_df.sample(frac=0.2, random_state=args.seed)
  
  log('Label proportions in training set: ' + compute_label_distribution(train_df, 'label', id2label, True))
  log('Label proportions in dev set: ' + compute_label_distribution(dev_df, 'label', id2label, True))
  log('Label proportions in test set: ' + compute_label_distribution(dev_df, 'label', id2label, True))
  
  # ----------------------------------------------------------------
  # - Prepare training ---------------------------------------------
  # ----------------------------------------------------------------
  
  cols = ['text', 'label']
  ecols = cols
  if args.id_col:
    ecols.append(args.id_col)
  if args.eval_by:
    ecols.append(args.eval_by)
  
  dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df[cols].reset_index(drop=True)),
    'dev': Dataset.from_pandas(dev_df[cols].reset_index(drop=True)),
    'test': Dataset.from_pandas(test_df[ecols].reset_index(drop=True)),
  })
  
  # load the tokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, max_length=256)
  
  # tokenize the data splits
  def tokenize(examples):
      return tokenizer(examples['text'], truncation=True, max_length=256)
  
  dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
  dataset.set_format("torch")
  
  # prepare the training arguments
  device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
  log(f"Using device '{str(device)}'")
  
  fp16_bool = True if args.fp16 and str(device) == 'cuda' else False
  
  if args.eval_metric:
    allowed_metrics = [
      'f1', 'balanced_accuracy', # for binary classification
      'f1_macro', 'balanced_accuracy_macro' # for multiclass classification
    ]
    if len(label2id) == 2:
      if args.eval_metric in allowed_metrics[:int(len(allowed_metrics)/2)]:
        eval_metric = args.eval_metric
      else:
        eval_metric = args.eval_metric.removesuffix('_macro')
        log(f"WARNING: value '{args.eval_metric}' passed to `--eval_metric` invalid for binary classification. Using '{eval_metric}' instead.")
    elif len(label2id) > 2:
      if args.eval_metric in allowed_metrics[int(len(allowed_metrics)/2):]:
        eval_metric = args.eval_metric
      else:
        eval_metric = args.eval_metric + '_macro'
        log(f"WARNING: value '{args.eval_metric}' passed to `--eval_metric` invalid for multiclass classification. Using '{eval_metric}' instead.")
    elif len(label2id) == 2:
      raise ValueError(f"invalid value '{args.eval_metric}' passed to `--eval_metric`. Allowed values for binary classification: {allowed_metrics[:int(len(allowed_metrics)/2)]}")
    else:
      raise ValueError(f"invalid value '{args.eval_metric}' passed to `--eval_metric`. Allowed values for multiclass classification: {allowed_metrics[int(len(allowed_metrics)/2):]}")
  else:
    eval_metric = 'f1_macro' if len(label2id) > 2 else 'f1'
  
  log(f"Using '{eval_metric}' as evaluation metric evaluation.")
  
  # create temporary directory for storing model checkpoints
  tmp_dir = tempfile.mkdtemp()
  
  train_args = TrainingArguments(
    # hyperparameters
    optim='adamw_torch',
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    warmup_ratio=args.warmup_ratio,
    weight_decay=args.weight_decay,
    per_device_train_batch_size=args.training_batch_size,
    per_device_eval_batch_size=args.training_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation,
  
    # reproducibility
    # full_determinism=args.train_deterministic, # see https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.full_determinism
    seed=args.seed if args.train_deterministic else 42, # 42 is default
    data_seed=args.seed if args.train_deterministic else None, # defaults to `seed`
    
    # model storing and loading
    logging_dir=os.path.join(tmp_dir, 'logs'),
    evaluation_strategy='epoch',
    report_to='all',
    output_dir=os.path.join(tmp_dir, 'results'),
    save_strategy='epoch',
    save_total_limit=2,
    save_safetensors=True,
    load_best_model_at_end=True,
    metric_for_best_model=eval_metric,
    greater_is_better=True,
    # efficiency
    fp16=fp16_bool,
    fp16_full_eval=False,
    # printing
    disable_tqdm=not args.train_verbose,
  )
  
  callbacks = []
  if args.early_stopping:
    callbacks.append(EarlyStoppingCallback(
      early_stopping_patience=args.early_stopping_patience,
      early_stopping_threshold=args.early_stopping_tolerance,
    ))
  
  if len(label2id) == 2:
    if args.downsample_train_data:
      pos_label = args.minority_label
    else:
      pos_label = id2label[0 if np.mean(dataset['train']['label'].numpy()) > 0 else 1]
    def compute_metrics(eval_pred):
      return compute_metrics_function(
          p=eval_pred,
          eval_fun=classification_metrics_binary,
          label2id=label2id,
          pos_label=pos_label,
        )
  else:
    def compute_metrics(eval_pred):
      return compute_metrics_function(
          p=eval_pred,
          eval_fun=classification_metrics_multiclass,
          label2id=label2id,
        )
  
  # load model
  clean_memory(device)
  log(f"Loading model '{args.model_name}'")
  model = AutoModelForSequenceClassification.from_pretrained(
      args.model_name,
      label2id=label2id,
      id2label=id2label,
      num_labels=len(label2id)
  )
  
  if args.class_weighting_strategy is None:
    m = len(label2id)
    class_weights = {i: 1/m for i in id2label.keys()}
  elif args.class_weighting_strategy == 'inverse_proportional':
    class_weights = compute_class_weights(dataset['train'])
  else:
    raise NotImplementedError(f'Value "{args.class_weighting_strategy}" passed to --class_weighting_strategy not implement. Set "inverse_proportional" or None.')
  
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  
  trainer = TrainerWithClassWeights(
    model=model,
    device=device,
    class_weights=class_weights,
    args=train_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['dev'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
  )
  if not args.train_verbose:
    trainer.remove_callback(PrinterCallback)
  
  # ----------------------------------------------------------------
  # - Train --------------------------------------------------------
  # ----------------------------------------------------------------
  
  log("Start finetuning")
  st = time.time()
  _ = trainer.train()
  training_time = time.time() - st
  
  # ----------------------------------------------------------------
  # - Evaluate -----------------------------------------------------
  # ----------------------------------------------------------------
  
  log("Start evaluation")
  st = time.time()
  out = evaluate(
    trainer=trainer,
    dataset=dataset['test'],
    id_col=args.id_col,
    eval_fun=classification_metrics_binary if len(label2id) == 2 else classification_metrics_multiclass,
    eval_by=args.eval_by,
    eval_by_values=args.eval_by_values.split(',') if args.eval_by_values else None,
    label2id=label2id,
    id2label=id2label,
    pos_label=pos_label if len(label2id) == 2 else None,
    seed=args.seed,
  )
  testing_time = time.time() - st
  
  out['duration'] = {'training': int(training_time), 'testing': int(testing_time)}
  
  log('Overall test set perfomances: '+', '.join([f"'{m}' = {v:.03f}" for m, v in out['metrics'].items() if 'f1' in m]))
  
  # add command line arguments (incl. defaults)
  out['args'] = vars(args)

  log(f"Writing results to '{args.out_file}'")
  with open(args.out_file, 'w') as f:
    json.dump(out, f, indent=2)
  
  if args.print_results:
    print('Overall results:')
    for m, vs in out['bootstrapped']['summarized']['overall'].items():
      print(f'{vs[0]:.03f} [{vs[1]:.03f}, {vs[2]:.03f}] {m}')

    if args.eval_by:
      print(f'\n{eval_metric.title()} by group:')
      grouped_eval = out['bootstrapped']['summarized']
      tab = pd.DataFrame({g: m[eval_metric] for g, m in grouped_eval.items()}).transpose()
      tab.columns = ['mean', 'q025', 'q975']
      print(tab.round(3))

  # ----------------------------------------------------------------
  # - Clean up -----------------------------------------------------
  # ----------------------------------------------------------------

  try:
    shutil.rmtree(tmp_dir)
    del df, train_df, dev_df, test_df, dataset, trainer
    model.cpu();
    del model, tokenizer, data_collator
    clean_memory(device)
    gc.collect()
  except:
    pass
  
if __name__ == '__main__':
  import argparse
  
  from datasets.utils.logging import disable_progress_bar
  disable_progress_bar()
  
  import transformers
  transformers.logging.set_verbosity_error()

  desc = [
    'Finetune Classifier',
    '',
    'This script allows to finetune a transformer-based classifier on a labeled text dataset.',
    'The test set results are written to a json file.',
  ]
  parser = argparse.ArgumentParser(description='\n'.join(desc), formatter_class=argparse.RawTextHelpFormatter)

  # Experiment arguments
  parser.add_argument('--experiment_name', type=str, default='dev', help='Name of the experiment')
  parser.add_argument('--out_file', type=str, default=None, help='Output file name')
  parser.add_argument('--out_file_overwrite', action='store_true', help='Overwrite output file if exists')
  parser.add_argument('--seed', type=int, default=1234, help='Random seed applied for data splitting and model training')
  parser.add_argument('--test_mode', action='store_true', help='Enable test mode')

  # Data preparation arguments
  parser.add_argument('--data_file', type=str, default='', help='Path to the data file. Should be a csv or tsv file.')
  parser.add_argument('--text_col', type=str, default='text', help='Name of the text column')
  parser.add_argument('--text_preprocessing', type=str, default=None, help='Text preprocessing method')
  parser.add_argument('--label_col', type=str, default='label', help='Name of the label column')
  parser.add_argument('--label_values', type=str, default=None, help='Label values to use (others are discarded). If not specified, all label values are used')
  parser.add_argument('--pos_label', type=str, default=None, help='Positive label')
  parser.add_argument('--id_col', type=str, default=None, help='Name of the column recording texts\' unique identifiers')
  parser.add_argument('--filter_by_col', type=str, default=None, help='Column to filter by')
  parser.add_argument('--filter_value', type=str, default=None, help='Value to filter by')

  # Data splitting arguments
  parser.add_argument('--sampling_strategy', type=str, default='random', help='Sampling strategy. Currently supported are "random" and "stratified" sampling.')
  parser.add_argument('--stratify_by_cols', type=str, default=None, help='Columns to stratify by if --sampling_strategy is "stratified"')
  parser.add_argument('--dev_size', type=float, default=0.2, help='Development set size')
  parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
  parser.add_argument('--downsample_train_data', action='store_true', help='Downsample training data')
  parser.add_argument('--downsample_minority_ratio', type=float, default=0.4, help='Ratio of minority class to majority class samples determining how many minority samples to keep')
  parser.add_argument('--minority_label', type=str, default=None, help='Label of the minority class')

  # Training arguments
  parser.add_argument('--model_name', type=str, default=None, help='Name of the model')
  parser.add_argument('--train_verbose', action='store_true', help='Printe training progress')
  parser.add_argument('--train_deterministic', action='store_true', help='Enable deterministic training')
  parser.add_argument('--class_weighting_strategy', type=str, default='inverse_proportional', help='Class weighting strategy. Currently only supports "inverse_proportional" (class weights inversely proportional to class prevalence in training data).')

  ## hyper parameters
  parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
  parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
  parser.add_argument('--training_batch_size', type=int, default=8, help='Training batch size')
  parser.add_argument('--gradient_accumulation', type=int, default=2, help='Gradient accumulation steps')
  parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio')
  parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
  parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
  parser.add_argument('--early_stopping_patience', type=int, default=3, help='Patience for early stopping')
  parser.add_argument('--early_stopping_tolerance', type=float, default=0.02, help='Tolerance for early stopping')
  parser.add_argument('--fp16', action='store_true', help='Enable mixed precision training')
  parser.add_argument('--save_model', action='store_true', help='Save the trained model')
  
  # evaluation
  parser.add_argument('--eval_metric', type=str, default='f1', help='Evaluation metric')
  parser.add_argument('--eval_by', type=str, default=None, help='Column by which valies to compute group-wise evaluation metrics in the test set')
  parser.add_argument('--eval_by_values', type=str, default=None, help='When `--eval_by` is specified, list the values for which evaluation metrics should be computed in a comma-separated string. If not set, all unique values in `--eval_by` column are used.')
  parser.add_argument('--print_results', action='store_true', help='Print evaluation results')

  args = parser.parse_args()
  
  ## Check arguments

  # check that output file is not already present
  if args.out_file is not None and not args.out_file_overwrite and os.path.isfile(args.out_file):
    raise ValueError(f'output file "{args.out_file}" already exists')
  
  # check that data file is present
  if not os.path.isfile(args.data_file):
    raise ValueError(f'input data file "{args.data_file}" not found')
  
  # check that text_preprocessing in allowed_preprocessing
  allowed_preprocessing = ['twitter']
  if args.text_preprocessing is not None and args.text_preprocessing not in allowed_preprocessing:
    raise ValueError(f'invalid value "{args.text_preprocessing}" passed to `--text_preprocessing`. Allowed values: {allowed_preprocessing}')
  
  # check that test_size is in [0, 1]
  if args.test_size < 0 or args.test_size > 1:
    raise ValueError(f'invalid value "{args.test_size}" passed to `--test_size`. Must be in [0, 1]')
  
  # check that dev_size is in [0, 1]
  if args.dev_size < 0 or args.dev_size > 1:
    raise ValueError(f'invalid value "{args.dev_size}" passed to `--dev_size`. Must be in [0, 1]')

  # check that downsample_minority_ratio is in [0, 1]
  if args.downsample_minority_ratio < 0 or args.downsample_minority_ratio > 1:
    raise ValueError(f'invalid value "{args.downsample_minority_ratio}" passed to `--downsample_minority_ratio`. Must be in [0, 1]')

  # check that sampling_strategy in allowed_sampling_strategies
  allowed_sampling_strategies = ['random', 'stratified']
  if args.sampling_strategy not in allowed_sampling_strategies:
    raise ValueError(f'invalid value "{args.sampling_strategy}" passed to `--sampling_strategy`. Allowed values: {allowed_sampling_strategies}')

  # raise error if model_name not specified
  if args.model_name is None:
    raise ValueError('`--model_name` not specified')
  
  # check that eval_metric in allowed_metrics
  allowed_metrics = ['f1', 'balanced_accuracy', 'f1_macro', 'balanced_accuracy_macro']
  if args.eval_metric not in allowed_metrics:
    raise ValueError(f'invalid value "{args.eval_metric}" passed to `--eval_metric`. Allowed values: {allowed_metrics}')

  # raise error if eval_by_values is specified but eval_by is not
  if args.eval_by_values is not None and args.eval_by is None:
    raise ValueError('`--eval_by_values` specified but `--eval_by` not')
  
  log(f"Starting experiment '{args.experiment_name}'")
  if args.out_file is None:
    args.out_file = args.experiment_name+'-results.json'
    log(f"WARNING: No output file specified. Writing to '{args.out_file}'")
  
  try:
    main(args)
  except Exception as e:
    log(f'ERROR: {str(e)}')
  else:
    log(f"Completed experiment '{args.experiment_name}'")
