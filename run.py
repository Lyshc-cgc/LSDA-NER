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
    cfg.training_args.output_dir = cfg.training_args.output_dir.format(dataset=cfg.dataset.dataset_name, seed=cfg.seed)
    cfg.training_args.seed = cfg.seed
    cfg.training_args.run_name = cfg.training_args.run_name.format(dataset=cfg.dataset.dataset_name, seed=cfg.seed)
    logger.info("*" * 20 + "Training Args" + "*" * 20)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("*" * 50)

    # 1. model and tokenizer
    best_ckpt_path = os.path.join(cfg.training_args.output_dir, 'best_ckpt')
    if not os.path.exists(best_ckpt_path):  # best ckpt does not exist
        logger.info('best checkpoint does not exist, load model from model_name_or_path')
        model_name_or_path = cfg.model_name_or_path
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:  # best ckpt exists
        logger.info(f'best checkpoint exists, load model from {best_ckpt_path}')
        model = AutoModelForSeq2SeqLM.from_pretrained(best_ckpt_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(best_ckpt_path)

    # 2. data
    # 2.1 pre-process
    processor = Processor(cfg.dataset, cfg.natural_label)
    dataset = processor.preprocess(k_shot=cfg.k_shot)
    original_columns = dataset['train'].column_names

    # 2.2 tokenize and padding
    def tokenize_pad_data(instances):
        """
        Tokenize data and padding data.
        :param instances: Instances to be processed. We tokenize the 'tokens' and 'tgt_sequence' columns.

        :return:
        """
        inputs = tokenizer(instances['tokens'], truncation=True, padding=True)
        labels = tokenizer(instances['tgt_sequence'], truncation=True, padding=True)
        inputs['labels'] = labels['input_ids']  # get tgt_sequence's input_ids as labels
        return inputs

    dataset = dataset.map(tokenize_pad_data, batched=True)
    dataset['validation'] = dataset['validation'].select(range(200))   # only use 200 samples for validation in training
    dataset['test'] = dataset['test'].select(range(200))  # only use 200 samples for testing

    # 2.3 get spans_labels, input_sents for metric computation
    spans_labels = {
        'train': dataset['train']['spans_labels'],
        'validation': dataset['validation']['spans_labels'],
        'test': dataset['test']['spans_labels']
    }
    input_sents = {
        'train': dataset['train']['tokens'],
        'validation': dataset['validation']['tokens'],
        'test': dataset['test']['tokens']
    }
    extra_data = {
        'span_labels': spans_labels,
        'input_sents': input_sents
    }
    # 2.4 remove original columns
    dataset = dataset.remove_columns(original_columns)

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
    logger.info('start evaluating...')
    eval_results = trainer.evaluate(eval_dataset=dataset['test'])
    logger.info('save the evaluation results...')
    trainer.log_metrics(split='test', metrics=eval_results)
    trainer.save_metrics(split='test', metrics=eval_results)

if __name__ == "__main__":
    main()