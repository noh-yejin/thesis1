import os
import time
import torch
import argparse # argument(매개변수에 집어넣는 값)를 사용하기 위한 내장함수

from model import SASRec
from utils import *

def str2bool(s): # 예외 처리
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true' 

parser = argparse.ArgumentParser() # argumentparer 인스턴스(parser 객체) 생성
# 사용할 인수 
# add_argument 메서드: argument에 대한 정의를 추가
# argument 이름/생략 가능 여부/argument가 없을 경우 사용 될 default값/argument의 타입
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True) # 필수
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool) #True/False
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args() # 사용자에게 전달받은 인수를 args에 저장

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    
f.close()
# import IPython; IPython.embed(colors="Linux"); exit(1) 
if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
 
    
    for name, param in model.named_parameters(): # named_parameters():(name, parameter)조합의 tuple iterator
        try:
            torch.nn.init.xavier_normal_(param.data) # xavier_normal_: 가중치를 다음과 같이 조정
                                                     # std=gain x 루트(2/(fan_in + fan_out))
                                                     # 주로 sigmoid 활성화 함수 앞에 사용
        except:
            pass # just ignore those failed init layers
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
        
    # 여기까지 두번째 Task
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            # load_state_dict:학습된 모델의 매개변수(state_dict)만 저장
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:] #불러온 모델의 파일 경로에서 epoch 다음 숫자를 찾음
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    

    if args.inference_only: # 추론 전용 모드에서 실행하는 코드
        model.eval() # 모델을 평가모드로 전환 
                     # dropout비활성화, 배치정규화의 이동 평균과 이동분산이 업데이트되지 않음.
        t_test = evaluate(model, dataset, args)  # evaluate에서 model.predict호출

        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    import IPython; IPython.embed(colors="Linux"); exit(1)

    # import IPython; IPython.embed(colors="Linux"); exit(1)
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()+sigmoid (이진 분류 문제에서 사용, output이 0~1사이의 값일 필요x)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98)) 
    # torch.optim.Adam는 경사하강법 알고리즘에서 Adam의 개선된 기능을 추가하여 최적화 수행
    # model.parameters(): 최적화할 모델의 파라미터 전달
    # lr(학습률) 기본값 0.001
    # betas: adam에서 사용되는 두 개의 모멘텀 계수(beta1, beta2)를 튜플 형태로 전달. 기본값은 (0.9,0.999)
 

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)  # forward 메서드
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            
              

            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0) # ignore padding
            loss = bce_criterion(pos_logits[indices], pos_labels[indices]) # 긍정적 예측 점수
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])  # 부정적 예측 점수
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
           
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
    
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    
    f.close()
    sampler.close()
    print("Done")
