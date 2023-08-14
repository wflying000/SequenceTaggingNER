import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import convert_logits_to_label_sequence


class EncoderSoftmaxTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        loss_calculator,
        metrics_calculator,
        output_dir,
        training_args,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_calculator = loss_calculator
        self.metrics_calculator = metrics_calculator
        self.output_dir = output_dir
        self.training_args = training_args
        self.early_stopping_count = 0
        self.writer = SummaryWriter(output_dir)
        self.device = None
        for _, p in model.named_parameters():
            self.device = p.device
            break
        

    def train(self):
        args = self.training_args
        num_train_batches = len(self.train_dataloader)

        best_weighted_f1 = -1
        best_micro_f1 = -1
        best_macro_f1 = -1

        best_weighted_p = -1
        best_micro_p = -1
        best_macro_p = -1

        best_weighted_r = -1
        best_micro_r = -1
        best_macro_r = -1

        for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):

            self.model.train()
            train_loss = 0
            self.metrics_calculator.clear()

            for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=num_train_batches, leave=False):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                outputs = self.model(batch)

                logits = outputs["logits"]
                labels = batch["labels"]

                if "loss" in outputs:
                    loss = outputs["loss"]
                else:
                    loss = self.loss_calculator(logits, labels)

                loss.backward()

                if (batch_idx + 1) % args.grad_accumulation_steps == 0 or (batch_idx + 1) == num_train_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()

                train_loss += loss.item()

                if "predictions" in outputs:
                    predictions = outputs["predictions"]
                    predictions = convert_logits_to_label_sequence(
                        logits=predictions, 
                        id2label=args.id2label, 
                        mask=batch["attention_mask"].detach().cpu(), 
                        add_special_tokens=args.add_special_tokens
                    )
                else:
                    logits = outputs["logits"]
                    predictions = convert_logits_to_label_sequence(
                        logits=logits, 
                        id2label=args.id2label, 
                        mask=batch["attention_mask"].detach().cpu(), 
                        add_special_tokens=args.add_special_tokens
                    )
                references = convert_logits_to_label_sequence(
                    logits=labels.detach().cpu(), 
                    id2label=args.id2label, 
                    mask=batch["attention_mask"].detach().cpu(), 
                    add_special_tokens=args.add_special_tokens
                )

                self.metrics_calculator.add_batch(predictions, references)
                
                global_step = num_train_batches * epoch + batch_idx
                if (global_step + 1) % args.write_step == 0:
                    self.writer.add_scalar("Loss/Step/Train", loss.item(), global_step=global_step)

                    if self.scheduler is not None:
                        cur_lr = self.scheduler.get_last_lr()[0]
                        self.writer.add_scalar("LearningRate-Step", cur_lr, global_step=global_step)
            
            train_metrics = self.metrics_calculator.compute()
            eval_result = self.evaluate()

            train_loss /= num_train_batches
            self.writer.add_scalar("Loss/Epoch/Train/", train_loss, global_step=epoch)
            train_category_metrics = train_metrics["category"]
            train_overall_metrics = train_metrics["overall"]
            for ent_type, metrics in train_category_metrics.items():
                for metric_type, value in metrics.items():
                    self.writer.add_scalar(f"Train-Entity/{ent_type}/{metric_type}", value, global_step=epoch)
            for metric_type, value in train_overall_metrics.items():
                self.writer.add_scalar(f"Train-Overall/{metric_type}", value, global_step=epoch)

            eval_loss = eval_result["loss"]
            eval_metrics = eval_result["metrics"]
            eval_category_metrics = eval_metrics["category"]
            eval_overall_metrics = eval_metrics["overall"]
            
            self.writer.add_scalar("Loss/Epoch/Eval/", eval_loss, global_step=epoch)
            for ent_type, metrics in eval_category_metrics.items():
                for metric_type, value in metrics.items():
                    self.writer.add_scalar(f"Eval-Entity/{ent_type}/{metric_type}", value, global_step=epoch)
            for metric_type, value in eval_overall_metrics.items():
                self.writer.add_scalar(f"Eval-Overall/{metric_type}", value, global_step=epoch)
            

            weighted_p = eval_overall_metrics["weighted-precision"]
            weighted_r = eval_overall_metrics["weighted-recall"]
            weighted_f1 = eval_overall_metrics["weighted-f1"]

            micro_p = eval_overall_metrics["micro-precision"]
            micro_r = eval_overall_metrics["micro-recall"]
            micro_f1 = eval_overall_metrics["micro-f1"]

            macro_p = eval_overall_metrics["macro-precision"]
            macro_r = eval_overall_metrics["macro-recall"]
            macro_f1 = eval_overall_metrics["macro-f1"]

            save_name = f"model_epoch_{epoch}_wf1_{weighted_f1:.4f}_wp_{weighted_p:.4f}_wr_{weighted_r:.4f}_mif1_{micro_f1:.4f}_mip_{micro_p:.4f}_mir_{micro_r:.4f}_maf1_{macro_f1:.4f}_map_{macro_p:.4f}_mar_{macro_r:.4f}.pth"
            save_path = os.path.join(self.output_dir, save_name)
            if hasattr(self.model, "module"):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            if self.training_args.stopping_metric_type == "precision":
                if (weighted_p > best_weighted_p) or (micro_p > best_micro_p) or (macro_p > best_macro_p):
                    
                    if weighted_p > best_weighted_p:
                        best_weighted_p = weighted_p
                    
                    if micro_p > best_micro_p:
                        best_micro_p = micro_p
                    
                    if macro_p > best_macro_p:
                        best_macro_p = macro_p
                    
                    self.early_stopping_count = 0
                    torch.save(state_dict, save_path)
                else:
                    self.early_stopping_count += 1
                    if self.early_stopping_count == self.training_args.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            elif self.training_args.stopping_metric_type == "recall":
                if (weighted_r > best_weighted_r) or (micro_r > best_micro_r) or (macro_r > best_macro_r):
                    if weighted_r > best_weighted_r:
                        best_weighted_r = weighted_r
                    if micro_r > best_micro_r:
                        best_micro_r = micro_r
                    if macro_r > best_macro_r:
                        best_macro_r = macro_r

                    self.early_stopping_count = 0
                    torch.save(state_dict, save_path)
                else:
                    self.early_stopping_count += 1
                    if self.early_stopping_count == self.training_args.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                if (weighted_f1 > best_weighted_f1) or (micro_f1 > best_micro_f1) or (macro_f1 > best_macro_f1):
                    if weighted_f1 > best_weighted_f1:
                        best_weighted_f1 = weighted_f1
                    if micro_f1 > best_micro_f1:
                        best_micro_f1 = micro_f1
                    if macro_f1 > best_macro_f1:
                        best_macro_f1 = macro_f1

                    self.early_stopping_count = 0
                    torch.save(state_dict, save_path)
                else:
                    self.early_stopping_count += 1
                    if self.early_stopping_count == self.training_args.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

    def evaluate(self):
        args = self.training_args
        self.model.eval()
        self.metrics_calculator.clear()

        eval_loss = 0

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), leave=False):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                outputs = self.model(batch)
                logits = outputs["logits"]
                labels = batch["labels"]

                if "loss" in outputs:
                    loss = outputs["loss"]
                else:    
                    loss = self.loss_calculator(logits, labels)

                eval_loss += loss.item()

                if "predictions" in outputs:
                    predictions = outputs["predictions"]
                    predictions = convert_logits_to_label_sequence(
                        logits=predictions, 
                        id2label=args.id2label, 
                        mask=batch["attention_mask"].detach().cpu(), 
                        add_special_tokens=args.add_special_tokens
                    )
                else:
                    logits = outputs["logits"]
                    predictions = convert_logits_to_label_sequence(
                        logits=logits, 
                        id2label=args.id2label, 
                        mask=batch["attention_mask"].detach().cpu(), 
                        add_special_tokens=args.add_special_tokens
                    )
                references = convert_logits_to_label_sequence(
                    logits=labels.detach().cpu(), 
                    id2label=args.id2label, 
                    mask=batch["attention_mask"].detach().cpu(), 
                    add_special_tokens=args.add_special_tokens
                )

                self.metrics_calculator.add_batch(predictions, references)
        
        eval_loss /= len(self.eval_dataloader)
        metrics = self.metrics_calculator.compute()

        eval_result = {
            "loss": eval_loss,
            "metrics": metrics,
        }

        return eval_result