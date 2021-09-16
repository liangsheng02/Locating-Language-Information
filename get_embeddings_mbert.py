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


def create_dl(dataset, batch_size=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_embeddings(dl, mode='cls', layer=12):
    """
    :param dl: dataloader
    :param mode: if 'cls': take the emb of 'cls' token;
                 if 'token': take every token.
    :param layer: 1-12
    :return:
    """
    embs = torch.Tensor(()).cpu()
    if mode == 'cls':
        for i in dl:
            b_input_ids = i["input_ids"].to(device)
            b_token_type_ids = i["token_type_ids"].to(device)
            b_attention_mask = i["attention_mask"].to(device)
            pooled_output = model(input_ids=b_input_ids,
                                  token_type_ids=b_token_type_ids,
                                  attention_mask=b_attention_mask)[0].detach().cpu()
            # We don't use pooled_output here, might be a problem?
            embs = torch.cat((embs, pooled_output))
            if embs.shape[0] >= 30000:
                return embs[:30000, :]
    elif mode == 'token':
        for i in dl:
            b_input_ids = i["input_ids"].to(device)
            b_token_type_ids = i["token_type_ids"].to(device)
            b_attention_mask = i["attention_mask"].to(device)
            hidden_states = model(input_ids=b_input_ids,
                                  token_type_ids=b_token_type_ids,
                                  attention_mask=b_attention_mask)[2][layer].detach().cpu()
            #(batch_size, sentence_length, 768)
            sentence_len = (i['attention_mask'] != 0).sum(dim=1) - 2 # un-pad & remove [cls][sep]
            for j in range(len(sentence_len)): # cat for per sentence
                emb = hidden_states[j, 1:sentence_len[j]+1, :]  # [sentence_len, 768]
                embs = torch.cat((embs, emb))
            if embs.shape[0] >= 30000:
                return embs[:30000, :]
    elif mode == 'mask':
        for i in dl:
            # randomly mask one token (between cls and sep)
            sentence_len = (i['attention_mask'] != 0).sum(dim=1) - 2
            b_input_ids = i["input_ids"].to(device)
            b_token_type_ids = i["token_type_ids"].to(device)
            b_attention_mask = i["attention_mask"].to(device)
            for mask_id in range(1, min(sentence_len)):
                # forward
                b_input_ids[:, mask_id] = torch.tensor(tokenizer.mask_token_id).to(device)
                hidden_states = model(input_ids=b_input_ids,
                                      token_type_ids=b_token_type_ids,
                                      attention_mask=b_attention_mask)[2][layer].detach().cpu()
                # cat masks
                embs = torch.cat((embs, hidden_states[:, mask_id, :]))
            if embs.shape[0] >= 10000:
                return embs[:10000, :]
    return embs[:30000] # (30000,768)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    args = parser.parse_args()

    dir_path = os.path.join('/mounts/work/language_subspace/mwiki_emb_2', args.mode, str(args.layer))
    if os.path.exists(os.path.join(dir_path, os.path.split(args.file)[1][:-4]+'.pt')):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists('/mounts/work/language_subspace/bert/model/'):
            model = transformers.BertModel.from_pretrained('/mounts/work/language_subspace/bert/model', output_hidden_states=True).to(device)
            tokenizer = transformers.BertTokenizer.from_pretrained('/mounts/work/language_subspace/bert/tokenizer')
        else:
            os.makedirs('/mounts/work/language_subspace/bert/')
            model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True).to(device)
            model.save_pretrained('/mounts/work/language_subspace/bert/model')
            tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            tokenizer.save_pretrained('/mounts/work/language_subspace/bert/tokenizer')
        # turn on eval mode
        model.eval()

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(args.file) as f:
            text = f.readlines()
        text = tokenizer(text, padding=True, truncation=True, pad_to_max_length=True)
        #print(len(text['input_ids']))
        text_dl = create_dl(create_ds(text))
        #print(create_ds(text).__len__())
        embs = get_embeddings(text_dl, args.mode, args.layer)
        if len(embs)< 30000:
            print(len(embs))
        output_file = os.path.join(dir_path, os.path.split(args.file)[1][:-4]+'.pt')
        torch.save(embs, output_file)
        print(output_file + ' saved.')
