import os
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,)
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
    logger.info(f"args:\n")
    logger.info("*" * 50)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("*" * 50)

    # 1. model and tokenizer
    model_name_or_path = cfg.model_name_or_path
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # 2. data
    # 2.1 pre-process
    processor = Processor(cfg.dataset, cfg.natural_label)
    dataset = processor.preprocess()
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

    dataset = dataset.map(tokenize_pad_data, batched=True)  # only use 200 samples for training
    dataset['train'] = dataset['train'].select(range(200))
    dataset['validation'] = dataset['validation'].select(range(200))
    dataset['test'] = dataset['test'].select(range(200))

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
        eval_dataset=dataset['train'],
        processing_class=tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(cfg.early_stopping_patience)]  # early stopping
    )
    trainer.setup_extra_for_metric(extra_data)

    # train or evaluate
    if cfg.train:
        trainer.train()
    trainer.evaluate(eval_dataset=dataset['train'])


if __name__ == "__main__":
    main()