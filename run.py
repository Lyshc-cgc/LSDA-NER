import json
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          EarlyStoppingCallback
                          )
from transformers.utils import logging
from src import NerTrainer, Processor, func_util

logger = logging.get_logger('run_script')

# load config
@hydra.main(version_base=None, config_path="./cfgs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 0. init
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    func_util.set_seed(cfg.seed)
    assert 'subset_size' in cfg and 'partition_time' in cfg, \
        '"subset_size" and "partition_time" should be provided in the kwargs for label subset partition.' \
        'Please check the specify "subset_size" and "partition_time" parameters in config file or command line.'

    if cfg.augmentation.startswith('lsp'):  # label subset partition
        cfg.augmentation += f'_size-{cfg.subset_size}'
        # if use negative sampling, add post fix '_neg' to the name of augmentation method
        if cfg.negative_portion > 0:  # negative_postfix
            cfg.augmentation += f'_neg-{cfg.negative_portion}'
        cfg.augmentation += f'_p-{cfg.partition_time}'  # partition times post fix
        cfg.training_args.output_dir = cfg.training_args.output_dir.format(
            aug_method=cfg.augmentation,
            k_shot=cfg.k_shot,
            dataset=cfg.dataset.dataset_name,
            seed=cfg.seed
        )  # ckpt and results path
        cfg.training_args.seed = cfg.seed  # set seed for training
        cfg.training_args.run_name = cfg.training_args.run_name.format(
            aug_method=cfg.augmentation,
            k_shot=cfg.k_shot,
            dataset=cfg.dataset.dataset_name,
            seed=cfg.seed
        )  # wandb run name
    else:  # if full-supervised or baseline
            cfg.training_args.output_dir = cfg.training_args.output_dir.format(
                aug_method=cfg.augmentation,
                dataset=cfg.dataset.dataset_name,
                seed=cfg.seed
            )  # ckpt and results path
            cfg.training_args.seed = cfg.seed  # set seed for training
            cfg.training_args.run_name = cfg.training_args.run_name.format(
                aug_method=cfg.augmentation,
                dataset=cfg.dataset.dataset_name,
                seed=cfg.seed
            )
    logger.info("*" * 20 + "Training Args" + "*" * 20)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("*" * 50)

    # 1. model and tokenizer
    best_ckpt_path = os.path.join(cfg.training_args.output_dir, 'best_ckpt')
    if not os.path.exists(best_ckpt_path):  # best ckpt does not exist
        logger.info('best checkpoint does not exist, load model from model_name_or_path for training from scratch...')
        model_name_or_path = cfg.model_name_or_path
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:  # best ckpt exists
        logger.info(f'best checkpoint exists, load model from {best_ckpt_path}')
        model = AutoModelForSeq2SeqLM.from_pretrained(best_ckpt_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(best_ckpt_path)

    # 2. data
    # 2.1 pre-process
    logger.info('start pre-processing...')
    processor = Processor(cfg, tokenizer)
    dataset, extra_data = processor.preprocess()
    logger.info('pre-processing finished.')
    logger.info(f'output path: {cfg.training_args.output_dir}')
    train_size = len(dataset['train'])
    # logger.info(f'train dataset size: {train_size}')

    dataset['validation'] = dataset['validation'].select(range(200))   # only use 200 samples for validation in training

    # 3. data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # 4. training args
    training_args = Seq2SeqTrainingArguments(**cfg.training_args)

    # 5. NerTrainer
    trainer = NerTrainer(
        id2label=processor.id2label,  # id2label and label2id are used for decoding in compute_metrics
        label2id=processor.label2id,
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(cfg.early_stopping_patience)]  # early stopping
    )
    trainer.setup_extra_for_metric(extra_data)

    # 6. if not exists best_ckpt_path, train and save the model
    if cfg.train and not os.path.exists(best_ckpt_path):
        logger.info('start training...')
        trainer.train()
        logger.info('training finished.')
        logger.info('save the best checkpoint...')
        trainer.save_model(best_ckpt_path)  # automatically save the best ckpt

    # 7. evaluate the best ckpt
    result_file = os.path.join(cfg.training_args.output_dir, 'all_results.json')
    if cfg.test and not os.path.exists(result_file):
        logger.info('start evaluating...')
        eval_results = trainer.evaluate(eval_dataset=dataset['test'])
        logger.info(f'save the evaluation results to {cfg.training_args.output_dir}')
        trainer.log_metrics(split='test', metrics=eval_results)
        trainer.save_metrics(split='test', metrics=eval_results)

        # 7.1 append the size of training dataset to the evaluation results
        with open(result_file, 'r') as f:
            data = json.load(f)
        data['eval_train_size'] = len(dataset['train'])
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()