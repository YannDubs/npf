#!/usr/bin/env bash

# Pretrain every img model! Remove `&` at the end if don't want parallel runs

logger="logs/train_imgs.out"
echo "STARTING" > $logger

#python bin/training/train_imgs.py -d celeba64 -m  GridedUnetCCP &
#python bin/training/train_imgs.py -d celeba64 -m  GridedSharedUnetCCP &
#python bin/training/train_imgs.py -d celeba64 -m  AttnCNP &

#python bin/training/train_imgs.py -d mnist -m  AttnCNP &
#python bin/training/train_imgs.py -d mnist -m  GridedSharedUnetCCP &
#python bin/training/train_imgs.py -d mnist -m  GridedCCP &

#wait

#python bin/training/train_imgs.py -d celeba32 -m  AttnCNP &
#python bin/training/train_imgs.py -d celeba32 -m  GridedUnetCCP &
#python bin/training/train_imgs.py -d celeba32 -m  GridedSharedUnetCCP &

#wait

python bin/training/train_imgs.py -d celeba32 -m  SelfAttnCNP &
python bin/training/train_imgs.py -d celeba32 -m  GridedCCP &

wait

python bin/training/train_imgs.py -d mnist -m  SelfAttnCNP &
#python bin/training/train_imgs.py -d mnist -m  GridedUnetCCP &

wait
