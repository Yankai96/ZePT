# set -x

# while true
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done
# echo $PORT
# srun -p smart_health_00 --gres=gpu:8 -N 1 --ntasks-per-node=1 --cpus-per-task=128 torchrun --nnodes 1 --nproc_per_node=8 --master_port $PORT train.py --dist True \
#     --num_workers 4 \
#     --num_samples 1 \
#     --batch_size 1 \
#     --log_name zept \
#     --store_num 20

srun -p smart_health_00 --gres=gpu:1 python test.py --pretrain_weights /mnt/petrelfs/huangzhongzhen/nips_code/ZePT/out/Part_ZePT_o/epoch_800.pth --save_dir test_res_o --store_result --dataset_list OVS
