conda activate vitis-ai-pytorch

# folders
export BUILD=./build
export LOG=${BUILD}/logs
export FLOAT_MODEL=./float_model
export QUANT_MODEL=./quant_model
mkdir -p ${LOG}
mkdir -p ${FLOAT_MODEL}

# run training
python Uformer/train.py --arch UNet --batch_size 32 --gpu '0' \
    --train_ps 128 --train_dir ./datasets/demoire/train --env 32_0705_1 \
    --val_dir ./datasets/demoire/val --embed_dim 32 --warmup

# copy models and logs 
cp Uformer/log/*.txt LOG/train.log
cp Uformer/log/models/model_best.pth FLOAT_MODEL/float_model.pth

# test the float model
python test.py --arch UNet --batch_size 32 --gpu '0' \
    --train_ps 128 --input_dir ./datasets/demoire/train \
    --result_dir ./results/demoire/ --embed_dim 32 2>&1 | tee ${LOG}/test_float.log




# quantize & export quantized model
python -u quantize.py -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log
python -u quantize.py -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log


# compile for target boards
source compile.sh zcu102 ${BUILD} ${LOG}
