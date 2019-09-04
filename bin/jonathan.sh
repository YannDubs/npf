for data in celeba32, celeba64, mnist
do
    for model in GridedCCP, SelfAttnCNP
    do
        ...
        ... task_id = ...
        ...
        python bin/training/train_imgs.py -d $data -m $model --starting-run $task_id \\
                                          --min-sigma 0.1 -e 1 --is-progressbar -b 64
    done
done

# 32x32 : 7Gb SelfAttn
# 32x32 : 2Gb celeba32
# multiply by to get celeba64
# 5 cpus

# SelfAttnCNP (16 batchsize)
# # MNIST : 6539 Gb
# # MNIST : 6539 Gb

python bin/training/train_imgs.py -d celeba64 -m SelfAttnCNP --starting-run $task_id \\
                                  --min-sigma 0.1 -e 1 --is-progressbar -b 16 -l 0.0005
python bin/training/train_imgs.py -d celeba64 -m GridedCCP --starting-run $task_id \\
                                --min-sigma 0.1 -e 1 --is-progressbar -b 64



# things that you can run only 1 (or 3) times as it is dev
python bin/training/train_imgs.py -d celeba32 -m GridedCCP --starting-run $task_id \\
                                  --min-sigma 0.1 -e 1 --is-progressbar -b 16 -l 0.0005

