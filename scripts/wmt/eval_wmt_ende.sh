#!/bin/bash

if [ $# -eq 0 ]
then
    echo "usage: $0 <data_dir>"
    exit 1
fi

DATA_PATH=$1
SP_PATH=/usr/local/bin

runconfig=/home/lgy/deepModels/tf_models/wmt_ende_transformer_4gpu_lr2_ws8000_dur2_0.998
#testset=newstest2017-ende
testset=newstest2014-deen
sl=en
tl=de

export PATH=$SP_PATH:$PATH
export CUDA_VISIBLE_DEVICES=7

wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/input-from-sgm.perl
wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

#perl input-from-sgm.perl < $DATA_PATH/test/$testset-src.$sl.sgm \
#   | spm_encode --model=data/wmt$sl$tl.model > data/$testset-src.$sl
#perl input-from-sgm.perl < $DATA_PATH/test/$testset-ref.$tl.sgm > data/$testset-ref.$tl


if false; then
  mkdir -p $runconfig/averaged
  onmt-average-checkpoints --max_count=10 \
                           --model_dir=$runconfig/ \
                           --output_dir=$runconfig/averaged/
fi

if false; then
  onmt-main infer \
            --model_type Transformer \
            --config config/wmt_ende.yml \
            --checkpoint_path=$runconfig/averaged \
            --features_file $DATA_PATH/$testset-src.$sl \
            > $DATA_PATH/$testset-src.hyp.$tl
fi
if true; then
  onmt-main infer \
            --model_type Transformer \
            --config config/wmt_ende.yml \
            --checkpoint_path=$runconfig/\
            --features_file $DATA_PATH/$testset-src.$sl \
            > $DATA_PATH/$testset-src.hyp.$tl
fi

if true; then
  ./spm_decode --model=$DATA_PATH/wmt$sl$tl.model --input_format=piece \
             < $DATA_PATH/$testset-src.hyp.$tl \
             > $DATA_PATH/$testset-src.hyp.detok.$tl

  perl multi-bleu-detok.perl $DATA_PATH/$testset-ref.$tl < $DATA_PATH/$testset-src.hyp.detok.$tl
fi
