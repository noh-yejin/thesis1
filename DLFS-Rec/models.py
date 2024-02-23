import torch
import torch.nn as nn
from modules import LayerNorm, Encoder, VanillaAttention


class DLFSRecModel(nn.Module):
    def __init__(self, args):
        super(DLFSRecModel, self).__init__()
        self.args = args
        self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # item 평균
        self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # item 공분산

        self.side_mean_dense = nn.Linear(args.feature_size, args.attribute_hidden_size) # SI 평균, 입력 벡터를 선형 변환하여 새로운 벡터를 생성
        self.side_cov_dense = nn.Linear(args.feature_size, args.attribute_hidden_size) # SI 공분산
        # nn.Embedding이 아니라 nn.Linear인 이유: item_attrs가 원핫인코딩 되어 있기 때문(nn.Embedding은 범주형 데이터를 다룰 때 사용)

        if args.fusion_type == 'concat': # side info+item 정보
            self.mean_fusion_layer = nn.Linear(args.attribute_hidden_size + args.hidden_size, args.hidden_size)
            self.cov_fusion_layer = nn.Linear(args.attribute_hidden_size + args.hidden_size, args.hidden_size)

        elif args.fusion_type == 'gate': # 'VanilaAttention'을 사용하여 게이트 메커니즘이 적용되는 퓨전레이어
            self.mean_fusion_layer = VanillaAttention(args.hidden_size, args.hidden_size)
            self.cov_fusion_layer = VanillaAttention(args.hidden_size, args.hidden_size)

        self.mean_layer_norm = LayerNorm(args.hidden_size, eps=1e-12) 
        self.cov_layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)
        self.elu = torch.nn.ELU()

        self.apply(self.init_weights) # 가중치 초기화

    def forward(self, input_ids, input_context):
        # 임베딩 계산
        mean_id_emb = self.item_mean_embeddings(input_ids)  # 아이템에 대한 평균 임베딩 [256,50,128]:[batch_size, max_seq_len, hiddien_size]
        cov_id_emb = self.item_cov_embeddings(input_ids)  # 아이템에 대한 공분산 임베딩 [256,50,128]

        input_attrs = self.args.items_feature[input_ids]    #input_ids를 이용하여 아이템의 추가 속성 가져옴[256, 50, 2326]:[batch_size, max_seq_len, hiddien_size]
        # side info 결합
        mean_side_dense = self.side_mean_dense(torch.cat((input_context, input_attrs), dim=2)) # side info에 대한 평균 계산 신경망/모듈
        cov_side_dense = self.side_cov_dense(torch.cat((input_context, input_attrs), dim=2)) # side info에 대한 공분산 계산 신경망/모듈

        if self.args.fusion_type == 'concat': # 임베딩을 이어붙여 새로운 시퀀스 임베딩을 만든다.
            mean_sequence_emb = self.mean_fusion_layer(torch.cat((mean_id_emb, mean_side_dense), dim=2)) 
            cov_sequence_emb = self.cov_fusion_layer(torch.cat((cov_id_emb, cov_side_dense), dim=2)) 
        elif self.args.fusion_type == 'gate': # gate메커니즘을 사용하여 임베딩 조절, 새로운 시퀀스 임베딩을 만든다.
            mean_concat = torch.cat(
                [mean_id_emb.unsqueeze(-2), mean_side_dense.unsqueeze(-2)], dim=-2)
            mean_sequence_emb, _ = self.mean_fusion_layer(mean_concat)
            cov_concat = torch.cat(
                [cov_id_emb.unsqueeze(-2), cov_side_dense.unsqueeze(-2)], dim=-2)
            cov_sequence_emb, _ = self.cov_fusion_layer(cov_concat)
        else:   # 두 임베딩을 단순히 더하여 새로운 시퀀스 임베딩을 만든다.
            mean_sequence_emb = mean_id_emb + mean_side_dense
            cov_sequence_emb = cov_id_emb + cov_side_dense

        mask = (input_ids > 0).long().unsqueeze(-1).expand_as(mean_sequence_emb) # 패딩 부분 제외하고 계산된 임베딩 적용
        mean_sequence_emb = mean_sequence_emb * mask  # 마스킹 처리
        cov_sequence_emb = cov_sequence_emb * mask

        mean_sequence_emb = self.dropout(self.mean_layer_norm(mean_sequence_emb)) # 드롭아웃
        cov_sequence_emb = self.elu(self.dropout(self.cov_layer_norm(cov_sequence_emb))) + 1 # 엘루(ELU)활성화 함수, 정규화 
        # 1을 더해서 공분산 속성이 양(+)의 값을 가지도록 조정

        item_encoded_layers = self.item_encoder(mean_sequence_emb,  
                                                cov_sequence_emb,
                                                output_all_encoded_layers=True) # 평균과 공분산 임베딩에 대해 인코딩 수행
        sequence_mean_output, sequence_cov_output = item_encoded_layers[-1]
        import IPython;IPython.embed(colors="Linux");exit(1)
        return sequence_mean_output, sequence_cov_output 

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
