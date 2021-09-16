from torch.utils.data import Dataset, DataLoader
import argparse, os
import torch
import transformers
import warnings
import random


class create_ds(Dataset):
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text['input_ids'])

    def __getitem__(self, item):
        return {key: torch.LongTensor(val[item]) for key, val in self.text.items()}


def create_dl(dataset, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_embeddings(dl, num=30000, n_layers=13):
    embs = [torch.Tensor(()).cpu() for _ in range(n_layers)]
    for i in dl:
        b_input_ids = i["input_ids"].to(device)
        #b_token_type_ids = i["token_type_ids"].to(device)
        b_attention_mask = i["attention_mask"].to(device)
        hidden_states = model(input_ids=b_input_ids,
                              #token_type_ids=b_token_type_ids,
                              attention_mask=b_attention_mask)[2]
        sentence_len = (i['attention_mask'] != 0).sum(dim=1) - 2  # un-pad & remove [cls][sep]
        for j in range(len(sentence_len)):
            for layer in range(n_layers):
                embs[layer] = torch.cat((embs[layer], hidden_states[layer].detach().cpu()[j, 1: sentence_len[j]-1, :]))
        if len(embs[0]) >= num:
            return embs
    return embs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--num', type=int, default=20000)
    args = parser.parse_args()

    if not os.path.exists(os.path.join('/mounts/work/language_subspace/cc-100_emb_2', 'token', '0', os.path.split(args.file)[1][:-4]+'.pt')):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists('/mounts/work/language_subspace/xlmr/model/'):
            model = transformers.XLMRobertaModel.from_pretrained('/mounts/work/language_subspace/xlmr/model', output_hidden_states=True).to(device)
            tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('/mounts/work/language_subspace/xlmr/tokenizer')
        else:
            os.makedirs('/mounts/work/language_subspace/xlmr/')
            model = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base', output_hidden_states=True).to(device)
            model.save_pretrained('/mounts/work/language_subspace/xlmr/model')
            tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
            tokenizer.save_pretrained('/mounts/work/language_subspace/xlmr/tokenizer')
        # turn on eval mode
        model.eval()
        with open(args.file) as f:
            text = f.readlines()
        text = tokenizer(text, padding=True, truncation=True, pad_to_max_length=True)
        text_dl = create_dl(create_ds(text))
        embs = get_embeddings(text_dl)
        for layer in range(13):
            dir_path = os.path.join('/mounts/work/language_subspace/cc-100_emb_2', 'token', str(layer))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            if len(embs[layer]) < 20000:
                print(len(embs[layer]))
            output_file = os.path.join(dir_path, os.path.split(args.file)[1][:-4]+'.pt')
            torch.save(embs[layer][:args.num], output_file)
            print(output_file + ' saved.')
