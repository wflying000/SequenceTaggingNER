import json
from typing import Any
import torch
from torch.utils.data import Dataset

def get_cluener_data(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_label2id(entity_types, scheme):
    labels = ["O"]
    if scheme == "BIOES":
        for ent_type in entity_types:
            if ent_type == "O":
                continue
            labels.append(f"B-{ent_type}")
            labels.append(f"I-{ent_type}")
            labels.append(f"E-{ent_type}")
            labels.append(f"S-{ent_type}")
    else:
        for ent_type in entity_types:
            if ent_type == "O":
                continue
            labels.append(f"B-{ent_type}")
            labels.append(f"I-{ent_type}")
    
    label2id = {label : idx for idx, label in enumerate(labels)}

    return label2id

class CLUENERDataset(Dataset):
    def __init__(
        self,
        data,
        label2id,
        tokenizer,
        add_special_tokens,
        max_length=512,
        tag_scheme="BIOES", 
    ):
        self.data = data
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        self.tag_scheme = tag_scheme
        self.special_tokens = list(tokenizer.special_tokens_map.values())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def collate_fn(self, item_list):
        text_list = [x["text"] for x in item_list]

        tokenized_text = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
            return_offsets_mapping=True,
        )

        input_ids = tokenized_text.input_ids
        bsz = len(item_list)
        seq_len = len(input_ids[0])

        labels = []
        for idx in range(bsz):
            cur_labels = ["O" for _ in range(seq_len)]
            annotation = item_list[idx]["label"]

            for ent_type, text2spans in annotation.items():
                for ent_text, spans in text2spans.items():
                    for span in spans:
                        # text[start_char_idx : end_char_idx + 1] == ent_text
                        start_char_idx = span[0]
                        end_char_idx = span[1]
                        token_idx_list = []
                        for char_idx in range(start_char_idx, end_char_idx + 1):
                            token_idx = tokenized_text.char_to_token(idx, char_idx)
                            if token_idx is not None:
                                token_idx_list.append(token_idx)
                        token_idx_list = sorted(list(set(token_idx_list)))

                        if self.tag_scheme == "BIOES":
                            if len(token_idx_list) == 1:
                                cur_labels[token_idx_list[0]] = f"S-{ent_type}"
                            else:
                                for j, token_idx in enumerate(token_idx_list):
                                    if j == 0:
                                        cur_labels[token_idx] = f"B-{ent_type}"
                                    elif j == len(token_idx_list) - 1:
                                        cur_labels[token_idx] = f"E-{ent_type}"
                                    else:
                                        cur_labels[token_idx] = f"I-{ent_type}"
                        else:
                            for j, token_idx in enumerate(token_idx_list):
                                if j == 0:
                                    cur_labels[token_idx] = f"B-{ent_type}"
                                else:
                                    cur_labels[token_idx] = f"I-{ent_type}"
            label_ids = []
            valid_length = sum(tokenized_text.attention_mask[idx])
            for j, (token, label) in enumerate(zip(tokenized_text.tokens(idx), cur_labels)):
                if (token in self.special_tokens) and \
                    ((self.add_special_tokens and (j == 0 or j >= valid_length - 1)) \
                        or (not self.add_special_tokens and j >= valid_length)):
                    label_ids.append(-100)
                else:
                    label_ids.append(self.label2id[label])
            labels.append(label_ids)
        
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.ByteTensor(tokenized_text.attention_mask)
        labels = torch.LongTensor(labels)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return batch

def test_dataset():
    import os, sys
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    os.chdir(sys.path[0])

    data_path = "../data/cluener/train.json"
    data = get_cluener_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained("../../pretrained_models/nghuyong/ernie-3.0-base-zh/")
    ent2id = json.load(open("../data/cluener/ent2id.json"))
    entity_types = [k for k, _ in ent2id.items()]
    scheme = "BIOES"
    label2id = get_label2id(entity_types, scheme)
    id2label = {v: k for k, v in label2id.items()}

    add_special_tokens = True

    dataset = CLUENERDataset(
        data,
        label2id,
        tokenizer,
        add_special_tokens,
        512,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    for batch in dataloader:
        labels = batch["labels"]
        
        for label in labels:
            label = label.tolist()
            tags = [id2label[x] if x != -100 else x for x in label]
            print(tags)

if __name__ == "__main__":
    test_dataset()

                        


        


        
        