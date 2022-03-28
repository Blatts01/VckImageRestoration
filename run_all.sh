#conda activate vitis-ai-pytorch

# folders
export BUILD=./build
export LOG=${BUILD}/logs
export FLOAT_MODEL=/workspace/float_model
export QUANT_MODEL=/workspace/quant_model
export QAT_MODEL=/workspace/qat/quant_model
export MODEL_RESULTS=/workspace/results/demoire/

mkdir -p ${LOG}
mkdir -p ${FLOAT_MODEL}
mkdir -p ${MODEL_RESULTS}
mkdir -p ${QAT_MODEL}

mkdir -p ${LOG}/model

# run training
#python Uformer/train.py --arch UNet --batch_size 32 --gpu '0' \
#    --train_ps 128 --train_dir /workspace/datasets/demoire/sidd/train/ --env 32_0705_1 \
#    --val_dir /workspace/datasets/demoire/sidd/val/ --embed_dim 32 --warmup

# copy models and logs 
#cp Uformer/log/UNet32_0705_1/*.txt ${LOG}/model/
#cp Uformer/log/UNet32_0705_1/models/model_best.pth ${FLOAT_MODEL}/float_model.pth

# test the float model

#python test.py --arch UNet --batch_size 32 --gpu '0' \
#    --train_ps 128 --input_dir /workspace/datasets/demoire/sidd/val/ \
#    --result_dir ${MODEL_RESULTS} --weights ${FLOAT_MODEL}/float_model.pth \
#    --embed_dim 32 2>&1 | tee ${LOG}/test_float.log

# quantize & export quantized model
#python -u quantize.py --quant_model_dir ${QUANT_MODEL} --quant_mode calib \
#    --input_dir /workspace/datasets/demoire/sidd/val/ \
#    --weights ${FLOAT_MODEL}/float_model.pth 2>&1 | tee ${LOG}/quant_calib.log

#python -u quantize.py --quant_model_dir ${QUANT_MODEL} --quant_mode test \
#    --input_dir /workspace/datasets/demoire/sidd/val/ \
#    --weights ${FLOAT_MODEL}/float_model.pth 2>&1 | tee ${LOG}/quant_test.log

# quantize with fast_finetune & export quantized model
#python -u quantize_fine.py --quant_model_dir ${QUANT_MODEL} --quant_mode calib \
#    --input_dir /workspace/datasets/demoire/sidd/val/ \
#    --weights ${FLOAT_MODEL}/float_model.pth 2>&1 | tee ${LOG}/quant_fast_calib.log

# qat training 
python -u qat.py --arch UNet --batch_size 32 --gpu '0' \
    --train_ps 128 --train_dir /workspace/datasets/demoire/sidd/train/ --env 32_0705_1 \
    --val_dir /workspace/datasets/demoire/sidd/val/ --embed_dim 32 --warmup \
    --save_dir ${QAT_MODEL} --qat_mode train

#python -u qat.py --arch UNet --batch_size 32 --gpu '0' \
#    --train_ps 128 --train_dir /workspace/datasets/demoire/sidd/train/ --env 32_0705_1 \
#    --val_dir /workspace/datasets/demoire/sidd/val/ --embed_dim 32 --warmup \
#    --save_dir ${QAT_MODEL} --qat_mode deploy

# compile for target boards
#source compile.sh zcu102 ${BUILD} ${LOG}
