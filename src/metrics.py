import numpy as np
from tqdm import tqdm
from typing import List
from seqeval.metrics import classification_report



class SeqTaggingMetrics():
    def __init__(self, id2label, decoder):
        self.id2label = id2label
        self.decoder = decoder
        entity_types = []
        for tag_id, tag in id2label.items():
            if tag == "O":
                continue
            ent_type = tag[2:]
            entity_types.append(ent_type)
        self.entity_types = entity_types

        self._init_category()
        
    
    def clear(self):
        self._init_category()
    
    def _init_category(self):
        self.category = self._create_category()

    def _create_category(self):
        category = {}
        for ent_type in self.entity_types:
            category[ent_type] = {}
            category[ent_type]["TP"] = 0
            category[ent_type]["FP"] = 0
            category[ent_type]["FN"] = 0
        return category
    
    
    def add_batch(self, predictions: List[List[str]], references: List[List[str]]):
        predictions = self.decoder.decode(predictions)
        references = self.decoder.decode(references)
        category = self._compute_category_count(predictions, references)
        self._update_category_count(category)
    
    def compute(self):
        category = self._compute_category_metrics(self.category)
        overall = self._compute_overall_metrics(category)
        metrics = {
            "category": category,
            "overall": overall,
        }
        return metrics

    def compute_metrics(self, predictions: List[List[str]], references: List[List[str]]):
        predictions = self.decoder.decode(predictions)
        references = self.decoder.decode(references)
        category = self._compute_category_count(predictions, references)
        category = self._compute_category_metrics(category)
        overall = self._compute_overall_metrics(category)
        metrics = {
            "category": category,
            "overall": overall,
        }
        return metrics

    
    def _compute_category_metrics(self, category):
        for key in category:
            TP = category[key]["TP"]
            FP = category[key]["FP"]
            FN = category[key]["FN"]
            category[key]["support"] = TP + FN

            precision = 0
            if TP + FP != 0:
                precision = TP / (TP + FP)
            
            recall = 0
            if TP + FN != 0:
                recall = TP / (TP + FN)
            
            f1 = 0
            if precision + recall != 0:
                f1 = (2 * precision * recall) / (precision + recall)
            
            category[key]["precision"] = precision
            category[key]["recall"] = recall
            category[key]["f1"] = f1
        
        return category

    def _compute_overall_metrics(self, category):
        TP = 0
        FP = 0
        FN = 0
        
        macro_precision = 0
        macro_recall = 0
        macro_f1 = 0

        weighted_precision = 0
        weighted_recall = 0
        weighted_f1 = 0

        for key in category:
            TP += category[key]["TP"]
            FP += category[key]["FP"]
            FN += category[key]["FN"]
            
            macro_precision += category[key]["precision"]
            macro_recall += category[key]["recall"]
            macro_f1 += category[key]["f1"]
            
            support = category[key]["TP"] + category[key]["FN"]
            weighted_precision += category[key]["precision"] * support
            weighted_recall += category[key]["recall"] * support
            weighted_f1 += category[key]["f1"] * support
        
        macro_precision /= len(category)
        macro_recall /= len(category)
        macro_f1 /= len(category)

        support = TP + FN
        if support != 0:
            weighted_precision /= support
            weighted_recall /= support
            weighted_f1 /= support

        micro_precision = 0
        if TP + FP != 0:
            micro_precision = TP / (TP + FP)
        
        micro_recall = 0
        if TP + FN != 0:
            micro_recall = TP / (TP + FN)
        
        micro_f1 = 0
        if micro_precision + micro_recall != 0:
            micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

        overall = {
            "micro-precision": micro_precision,
            "micro-recall": micro_recall,
            "micro-f1": micro_f1,
            "macro-precision": macro_precision,
            "macro-recall": macro_recall,
            "macro-f1": macro_f1,
            "weighted-precision": weighted_precision,
            "weighted-recall": weighted_recall,
            "weighted-f1": weighted_f1,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "support": support,
            "num_pred": TP + FP,
        }

        return overall
    
    def _update_category_count(self, category):
        for key in category:
            self.category[key]["TP"] += category[key]["TP"]
            self.category[key]["FP"] += category[key]["FP"]
            self.category[key]["FN"] += category[key]["FN"]
    
    def _compute_category_count(self, predictions, references):
        category = self._create_category()
        new_predictions = []
        for idx, preds in enumerate(predictions):
            for span in preds:
                new_span = (idx,) + span
                new_predictions.append(new_span)

        new_references = []        
        for idx, refs in enumerate(references):
            for span in refs:
                new_span = (idx,) + span
                new_references.append(new_span)
        
        pred_set = set(new_predictions)
        true_set = set(new_references)

        tp_set = pred_set.intersection(true_set)
        fp_set = pred_set.difference(true_set)
        fn_set = true_set.difference(pred_set)

        for span in tp_set:
            ent_type = span[-1]
            category[ent_type]["TP"] += 1
        for span in fp_set:
            ent_type = span[-1]
            category[ent_type]["FP"] += 1
        for span in fn_set:
            ent_type = span[-1]
            category[ent_type]["FN"] += 1
        
        return category
    

def test():
    from utils import SequenceTaggingDecoder

    id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-MISC", 4: "I-MISC"}

    decoder = SequenceTaggingDecoder(scheme="BIO")
    metrics = SeqTaggingMetrics(id2label, decoder)

    y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]

    res1 = classification_report(y_true, y_pred, output_dict=True)

    res2 = metrics.compute_metrics(y_pred, y_true)

    print(res1)

    print(res2)

if __name__ == "__main__":
    test()




    