python train_blur.py \
	--train_file ./dataset_make/cycle_train.h5 \
	--valid_file ./dataset_make/cycle_valid.h5 \
	--niter 10 \
	--batch_size 2 \
	--output_dir .

python train_deblur.py \
	--train_file ./dataset_make/deblur_train.h5 \
	--valid_file ./dataset_make/deblur_valid.h5 \
	--niter 5 \
	--batch_size 8 \
	--blur_model_path ./models/bgan_generator.pth \
	--output_dir .
