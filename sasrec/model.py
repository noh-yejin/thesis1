import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs 

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args): #user_num,item_num,args를 입력으로 받아 모델을 초기화
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # 임베딩 층, Layer Normalization, Multihead Attention, PointWise FeedForward 등을 정의하고 초기화
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()  # 모듈을 리스트 형태로 저장
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # sequential 데이터를 입력으로 받아 임베딩
        # 1.Item Embedding 
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        # 2.Position Embedding
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # 3.Embedding Dropout        
        seqs = self.emb_dropout(seqs)
        # 4.Masking the padding elements in the sequence
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev) # 현재 시간 이후의 정보를 마스킹
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        # 5.creating attention mask for enforcing causality
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # 6.Self-Attention and FeedForward Layers
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        # 7.Layer Normalization for the final features
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        
        return log_feats
    # 모델의 순방향 전달 메서드, 학습 시 사용
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # positive item embedding
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) # negative item embedding

        pos_logits = (log_feats * pos_embs).sum(dim=-1) # 로짓을 이용하여 손실계산, 차원축소
        neg_logits = (log_feats * neg_embs).sum(dim=-1) 

        # pos_logits-실제 관람한 영화정보, 사용자가 상호작용한 아이템의 시퀀스를 고려하여 다음에 추천할 영화가 얼마나 관련이 있는지를 의미함->관련성 높은 아이템을 추천하려고 할 것.
        # 값이 클 수록 관련성이 높으며 log loss값이 최소가 됨.
        # neg_logits-부정적인 관련성, 사용자가 관심을 가질 가능성이 낮은 아이템들에 대한 것. 점수가 낮을수록 해당 아이템이 사용자의 이전 행동과 관련이 낮다는 걸 의미.

        # pos_pred = self.pos_sigmoid(pos_logits) -> 이 값이 커지고
        # neg_pred = self.neg_sigmoid(neg_logits) -> 이 값이 작아져야 log_loss가 최소가 된다.
        
        return pos_logits, neg_logits # pos_pred, neg_pred

    # 다음 시점에서 관람할 영화를 예측
    def predict(self, user_ids, log_seqs, item_indices): # item_indices:유저가 다음에 관람할 영화 ID(ground truth)+유저가 관람하지 않은 영화 100개로 이루어진 벡터
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        # log2feats 메서드를 사용하여 아이템 임베딩과의 내적 계산, 각 아이템에 대한 로짓을 얻음

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        # 1xd(50)
        import IPython; IPython.embed(colors="Linux"); exit(1)  

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        # item embedding layer->101 X d 차원의 텐서로 변환 
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        # unsqueeze(-1): 특성 벡터를 열 벡터로 변환/squeeze(-1): 차원이 1인 차원을 제거

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
        # output:101 x 1 차원
        # 해당 벡터에서 가장 높은 값을 가진 인덱스에 해당하는 영화=유저가 다음 시점에 관람할 영화로 예측
