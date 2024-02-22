#!/bin/bash
for hidden_units in 10 50
do
    python -u main.py \
        --device=cpu \
        --dataset='test'\
        --num_epochs=3\
        --train_dir='test_modcloth'>result/$hidden_units'_test1'.log 
        
        # 결과를 result폴더 안에 저장. 변수명은 보통 특징 나열해서 작성
done
