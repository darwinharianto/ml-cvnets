# convert
export PYTHONPATH=.
export CONFIG_FILE=/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/config/segmentation/labelme/pspnet_mobilevitv2.yaml
export N_CLASSES=6
export MODEL_WEIGHTS=/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/pspnet_mobilevitv2_results/width_1_0_0/train/checkpoint_last.pt
cvnets-convert --common.config-file $CONFIG_FILE --common.results-loc results_converted/coreml_models_res --model.segmentation.pretrained $MODEL_WEIGHTS  --conversion.coreml-extn mlmodel --model.segmentation.n-classes $N_CLASSES

# train bolt all
export PYTHONPATH=.
export CONFIG_FILE=/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/config/segmentation/labelme/pspnet_mobilevitv2.yaml
export N_CLASSES=6
export PRETRAINED_WEIGHTS=model_zoo/mobilevitv2-1.0.pt
PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/segmentation/labelme/pspnet_mobilevitv2.yaml --common.results-loc exp_results/pspnet_mobilevitv2_results/width_1_0_0 --common.override-kwargs model.classification.pretrained='/Users/darwinharianto/Downloads/mobilevitv2-1.0.pt'

# train bolt single
export PYTHONPATH=.
export CONFIG_FILE=/Users/darwinharianto/Documents/Git/git_training_lab/python/ml-cvnets/config/segmentation/labelme/pspnet_mobilevitv2.yaml
export N_CLASSES=6
export PRETRAINED_WEIGHTS=model_zoo/mobilevitv2-1.0.pt
PYTHONWARNINGS="ignore" cvnets-train --common.config-file config/segmentation/labelme/pspnet_mobilevitv2_single.yaml --common.results-loc exp_results/pspnet_mobilevitv2_single_results/width_1_0_0 --common.override-kwargs model.classification.pretrained=$PRETRAINED_WEIGHTS