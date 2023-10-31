# Author: Noah Jones

import logging
import os
from statistics import mean, stdev
import sys
import pdb
import pandas as pd
from typing import Callable, Dict
import numpy as np
from pprint import pformat
from scipy.special import softmax
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    Trainer,
    EvalPrediction,
    set_seed
)

from multimodal_exp_args import MultimodalDataTrainingArguments, ModelArguments, OurTrainingArguments
from evaluation import calc_classification_metrics, calc_regression_metrics
from multimodal_transformers.data import load_data_from_folder, load_data_into_folds
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular
from util import create_dir_if_not_exists, get_args_info_as_str

os.environ['COMET_MODE'] = 'DISABLED'
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, MultimodalDataTrainingArguments,
                               OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    

    # Setup logging
    create_dir_if_not_exists(training_args.output_dir)
    stream_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(filename=os.path.join(training_args.output_dir, 'train_log.txt'),
                                       mode='w+')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        # level=logging.WARN,
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[stream_handler, file_handler]
    )

    logger.info(f"======== Model Args ========\n{get_args_info_as_str(model_args)}\n")
    logger.info(f"======== Data Args ========\n{get_args_info_as_str(data_args)}\n")
    logger.info(f"======== Training Args ========\n{get_args_info_as_str(training_args)}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if not data_args.create_folds:
        train_dataset, val_dataset, test_dataset = load_data_from_folder(
            data_args.data_path,
            data_args.column_info['text_cols'],
            tokenizer,
            label_col=data_args.column_info['label_col'],
            label_list=data_args.column_info['label_list'],
            categorical_cols=data_args.column_info['cat_cols'],
            numerical_cols=data_args.column_info['num_cols'],
            categorical_encode_type=data_args.categorical_encode_type,
            numerical_transformer_method=data_args.numerical_transformer_method,
            sep_text_token_str=tokenizer.sep_token if not data_args.column_info['text_col_sep_token'] else data_args.column_info['text_col_sep_token'],
            max_token_length=training_args.max_token_length,
            debug=training_args.debug_dataset,
        )
        train_datasets = [train_dataset]
        val_datasets = [val_dataset]
        test_datasets = [test_dataset]
    else:
        train_datasets, val_datasets, test_datasets = load_data_into_folds(
            data_args.data_path,
            data_args.num_folds,
            data_args.validation_ratio,
            data_args.column_info['text_cols'],
            tokenizer,
            label_col=data_args.column_info['label_col'],
            label_list=data_args.column_info['label_list'],
            categorical_cols=data_args.column_info['cat_cols'],
            numerical_cols=data_args.column_info['num_cols'],
            categorical_encode_type=data_args.categorical_encode_type,
            numerical_transformer_method=data_args.numerical_transformer_method,
            sep_text_token_str=tokenizer.sep_token if not data_args.column_info['text_col_sep_token'] else
            data_args.column_info['text_col_sep_token'],
            max_token_length=training_args.max_token_length,
            debug=training_args.debug_dataset,
        )
    train_dataset = train_datasets[0]

    set_seed(training_args.seed)
    task = data_args.task
    if task == 'regression':
        num_labels = 1
    else:
        num_labels = len(np.unique(train_dataset.labels)) if data_args.num_classes == -1 else data_args.num_classes

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if task_name == "classification":
                preds_labels = np.argmax(p.predictions, axis=1)
                if p.predictions.shape[-1] == 2:
                    pred_scores = softmax(p.predictions, axis=1)[:, 1]
                else:
                    pred_scores = softmax(p.predictions, axis=1)
                return calc_classification_metrics(pred_scores, preds_labels,
                                                   p.label_ids)
            elif task_name == "regression":
                preds = np.squeeze(p.predictions)
                return calc_regression_metrics(preds, p.label_ids)
            else:
                return {}
        return compute_metrics_fn
        
    eval_results = {}
    total_results = []
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(zip(train_datasets, val_datasets, test_datasets)):
        logger.info(f'======== Fold {i+1} ========')
        
        if data_args.mode != "infer":
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
            )
            tabular_config = TabularConfig(num_labels=num_labels,
                                        cat_feat_dim=train_dataset.cat_feats.shape[
                                            1] if train_dataset.cat_feats is not None else 0,
                                        numerical_feat_dim=train_dataset.numerical_feats.shape[
                                            1] if train_dataset.numerical_feats is not None else 0,
                                        **vars(data_args))
            config.tabular_config = tabular_config


            #need to initialize model this way for hyperparameter tuning
            def model_init():
                    return AutoModelWithTabular.from_pretrained(
                    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                    config=config,
                    cache_dir=model_args.cache_dir
                )
            if i == 0:
                logger.info(tabular_config)
                logger.info(model_init())

            if data_args.oversample:
                import torch
                from torch.utils.data import WeightedRandomSampler, DataLoader
                from transformers import Trainer
                class CW_Trainer(Trainer):
                    def get_train_dataloader(self):
                        """
                        Returns the training :class:`~torch.utils.data.DataLoader`.

                        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
                        to distributed training if necessary) otherwise.

                        Subclass and override this method if you want to inject some custom behavior.
                        """
                        if self.train_dataset is None:
                            raise ValueError("Trainer: training requires a train_dataset.")
                        # train_sampler = self._get_train_sampler()
                        target = np.array(train_dataset.labels)
                        print('target train 0/1: {}/{}'.format(
                            len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
                        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
                        weight = 1. / class_sample_count
                        target = target.astype("int64")
                        samples_weight = np.array([weight[t] for t in target])
                        samples_weight = torch.from_numpy(samples_weight)
                        samples_weight = samples_weight.double()
                        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


                        return DataLoader(
                        self.train_dataset,
                        batch_size=self.args.train_batch_size,
                        sampler=train_sampler,
                        collate_fn=self.data_collator,
                        drop_last=self.args.dataloader_drop_last,
                        )
                trainer = CW_Trainer(
                    model_init=model_init,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=build_compute_metrics_fn(task),
                    )   
            else:
                trainer = Trainer(
                    model_init=model_init,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=build_compute_metrics_fn(task),
                )


        #For tuning all of the hyperparameters
        if data_args.mode=='hyp_tune':
            import torch
            from torch.utils.data import WeightedRandomSampler, DataLoader
            from transformers import Trainer

            def my_hp_space(trial) -> Dict[str, float]:
                #optuna 
                return {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 9e-5, log=False),
                    "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 4),
                    "weight_decay": trial.suggest_categorical("weight_decay", [0, 2]),
                    "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16]),
                    "save_steps": trial.suggest_categorical("save_steps", [20000]),

                }

            best_trial = trainer.hyperparameter_search(
            direction="maximize", 
            backend="optuna",
            hp_space=my_hp_space, 
            n_trials=10)
            print(best_trial)
            for n, v in best_trial.hyperparameters.items():
                setattr(trainer.args, n, v)

            trainer.train()
            trainer.save_model()   
            tokenizer.save_pretrained(training_args.output_dir)

        if data_args.mode=='train':
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)

        # Evaluation
        if training_args.do_eval and data_args.mode!="infer":
            logger.info("*** Evaluate ***")
            eval_result = trainer.evaluate(eval_dataset=val_dataset)
            logger.info(pformat(eval_result, indent=4))

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_metric_results_{task}_fold_{i+1}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(task))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

        if training_args.do_predict:

            from transformers import Trainer
            import torch
            from dataclasses import replace
            logging.info("*** Test ***")
            if data_args.mode != 'infer':
                pass
            else:
                test_model = AutoModelWithTabular.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path)
                # args = torch.load(f"{model_args.model_name_or_path}/training_args.bin")
                # # args_ = replace(args, no_cuda=True)
                # can add these args to the trainer if needed
                trainer = Trainer(model=test_model,compute_metrics=build_compute_metrics_fn(task))
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{task}_seed_{training_args.seed}_fold_{i+1}.txt"
            )
            eval_result = trainer.evaluate(eval_dataset=test_dataset)
            logger.info(pformat(eval_result, indent=4))
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {}  *****".format(task))
                    # writer.write("index\tSITB_PRESENT(=1)\tprediction\n")
                    if task == "classification":
                        output_pred_file = os.path.join(
                        training_args.output_dir, f"test_results_{task}_seed_{training_args.seed}_fold_{i+1}.csv"
                        )
                        prediction_labels = np.argmax(predictions, axis=1)
                        s_present_predictions = predictions[:,1]
                        s_absent_predictions = predictions[:,0]
                        test_results = pd.DataFrame({'SITB_PRESENT(=1)':s_present_predictions, 'SITB_ABSENT(=0)':s_absent_predictions,'label':prediction_labels})
                        test_results.to_csv(output_test_file,index=False)

                    for index, item in enumerate(prediction_labels):
                        if task == "regression":
                            writer.write("%d\t%3.3f\t%d\n" % (index, item, test_dataset.labels[index]))
                        else:
                            pass
                            #pass did this up there below is the original (NCJ)
                            # item = test_dataset.get_labels()[item]
                            # val = predictions[index]
                            # writer.write("%d\t%s\t%3.3f\n" % (index, item, val))
                
                output_test_file = os.path.join(
                    training_args.output_dir, f"test_metric_results_{task}_fold_{i+1}.txt"
                )
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(task))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                eval_results.update(eval_result)
        
        del trainer
        if data_args.mode != 'infer':
            del config
            del tabular_config

        torch.cuda.empty_cache()
        total_results.append(eval_results)
    aggr_res = aggregate_results(total_results)
    logger.info('========= Aggr Results ========')
    logger.info(pformat(aggr_res, indent=4))

    output_aggre_test_file = os.path.join(
        training_args.output_dir, f"all_test_metric_results_{task}.txt"
    )
    with open(output_aggre_test_file, "w") as writer:
        logger.info("***** Aggr results {} *****".format(task))
        for key, value in aggr_res.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))


def aggregate_results(total_test_results):
    metric_keys = list(total_test_results[0].keys())
    aggr_results = dict()

    for metric_name in metric_keys:
        if type(total_test_results[0][metric_name]) is str:
            continue
        res_list = []
        for results in total_test_results:
            res_list.append(results[metric_name])
        if len(res_list) == 1:
            metric_avg = res_list[0]
            metric_stdev = 0
        else:
            metric_avg = mean(res_list)
            metric_stdev = stdev(res_list)

        aggr_results[metric_name + '_mean'] = metric_avg
        aggr_results[metric_name + '_stdev'] = metric_stdev
    return aggr_results


if __name__ == '__main__':
    main()
