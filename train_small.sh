#set -aux
NITER=100
BATCH_SIZE=42
begin_time=`date`
echo "Begin to train how to blur: $begin_time"
python -W ignore train_blur.py \
	--train_file cycle_train.tfrecord \
	--valid_file cycle_valid.tfrecord \
	--niter ${NITER} \
	--batch_size ${BATCH_SIZE} \
	--output_dir bgan \
	--lr 3e-5
finish_time=$(date)
echo "Finish: $finish_time \n"

mv bgan/bgan_generator_snapshot_99.pth bgan_pretrain.pth
mv bgan/dbgan_generator_snapshot_99.pth dbgan_pretrain.pth

begin_time=`date`
echo "Begin to train how to deblur: $begin_time"
python -W ignore train_deblur.py \
	--train_file deblur_train.tfrecord \
	--valid_file deblur_valid.tfrecord \
	--niter ${NITER} \
	--batch_size ${BATCH_SIZE} \
	--blur_model_path bgan_pretrain.pth \
	--output_dir dbgan \
	--lr 3e-5
finish_time=$(date)
echo "Finish: $finish_time \n"

mv dbgan/dbgan_generator_snapshot_0.pth dbgan_pretrain.pth
