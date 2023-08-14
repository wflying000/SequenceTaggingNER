import os
import json
import time
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from loss import SeqTaggingNerLoss
from metrics import SeqTaggingMetrics
from trainer import EncoderSoftmaxTrainer
from optimization import get_optimizer
from model import EncoderSoftmaxForNer, EncoderCRFForNer
from dataset import CLUENERDataset, get_cluener_data, get_label2id
from utils import SequenceTaggingDecoder, GeneralConfig


def get_args():
    args_parser = ArgumentParser()
    args_parser.add_argument("--decoder", type=str, default="crf")
    args_parser.add_argument("--ent2id_path", type=str, default="../data/cluener/ent2id.json")
    args_parser.add_argument("--tag_scheme", type=str, default="BIOES")
    args_parser.add_argument("--train_data_path", type=str, default="../data/cluener/train.json")
    args_parser.add_argument("--eval_data_path", type=str, default="../data/cluener/dev.json")
    args_parser.add_argument("--pretrained_model_path", type=str, default="../../pretrained_model/ernie-3.0-base-zh/")
    args_parser.add_argument("--add_special_tokens", type=bool, default=True)
    args_parser.add_argument("--max_length", type=int, default=512)
    args_parser.add_argument("--train_batch_size", type=int, default=32)
    args_parser.add_argument("--eval_batch_size", type=int, default=32)
    args_parser.add_argument("--dropout", type=float, default=0.1)
    args_parser.add_argument("--device", type=str, default="cuda")
    args_parser.add_argument("--learning_rate", type=float, default=5e-5)
    args_parser.add_argument("--weight_decay", type=float, default=0.01)
    args_parser.add_argument("--num_epochs", type=int, default=20)
    args_parser.add_argument("--write_step", type=int, default=1)
    args_parser.add_argument("--patience", type=int, default=5)
    args_parser.add_argument("--stopping_metric_type", type=str, default="precision")
    args_parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    args_parser.add_argument("--output_dir", type=str, default="../outputs/")

    args = args_parser.parse_args()

    return args


def train(args):

    ent2id = json.load(open(args.ent2id_path))
    entity_types = [k for k, _ in ent2id.items()]
    label2id = get_label2id(entity_types, args.tag_scheme)
    id2label = {v: k for k, v in label2id.items()}

    train_data = get_cluener_data(args.train_data_path)
    eval_data = get_cluener_data(args.eval_data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    pretrained_model_config = AutoConfig.from_pretrained(args.pretrained_model_path)

    train_dataset = CLUENERDataset(
        data=train_data,
        label2id=label2id,
        tokenizer=tokenizer,
        add_special_tokens=args.add_special_tokens,
        max_length=args.max_length,
        tag_scheme=args.tag_scheme,
    )

    eval_dataset = CLUENERDataset(
        data=eval_data,
        label2id=label2id,
        tokenizer=tokenizer,
        add_special_tokens=args.add_special_tokens,
        max_length=args.max_length,
        tag_scheme=args.tag_scheme,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
    )

    model_config = GeneralConfig(
        pretrained_model_path=args.pretrained_model_path,
        hidden_size=pretrained_model_config.hidden_size,
        num_labels=len(label2id),
        dropout=args.dropout,
    )

    if args.decoder == "softmax":
        model = EncoderSoftmaxForNer(model_config)
    elif args.decoder == "crf":
        model = EncoderCRFForNer(model_config)
    device = torch.device(args.device)
    model = model.to(device)

    optimizer = get_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None

    training_args = GeneralConfig(
        num_epochs=args.num_epochs,
        write_step=args.write_step,
        patience=args.patience,
        stopping_metric_type=args.stopping_metric_type,
        grad_accumulation_steps=args.grad_accumulation_steps,
        id2label=id2label,
        add_special_tokens=args.add_special_tokens,
        tokenizer=tokenizer,
    )

    tag_decoder = SequenceTaggingDecoder(args.tag_scheme)
    metrics_caculator = SeqTaggingMetrics(
        id2label=id2label,
        decoder=tag_decoder,
    )

    loss_calculator = SeqTaggingNerLoss()

    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    output_dir = f"{args.output_dir}/{now_time}/"
    os.makedirs(output_dir)

    trainer = EncoderSoftmaxTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_calculator=loss_calculator,
        metrics_calculator=metrics_caculator,
        output_dir=output_dir,
        training_args=training_args,
    )

    trainer.train()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    import sys
    os.chdir(sys.path[0])
    main()




