#python train_blur.py \
#	--train_file ./cycle_dataset_small/cycle_train.h5 \
#	--valid_file ./cycle_dataset_small/cycle_valid.h5 \
#	--niter 10 \
#	--batch_size 4\

python train_deblur.py \
	--train_file ./deblur_train.h5 \
	--valid_file ./deblur_valid.h5 \
	--niter 5 \
	--batch_size 16 \
	--blur_model_path ./models/bgan_generator.pth \
	--output_dir .
