data/data-train.hdf5: data/train-source.txt
	cd seq2seq-attn && python preprocess.py \
		--srcfile data/train-source.txt \
		--targetfile data/train-target.txt \
		--srcvalfile data/validation-source.txt \
		--targetvalfile data/validation-target.txt \
		--outputfile data/data

data/train-source.txt:
	th prepreprocess.lua --srcfile data/train-source.txt --targetfile data/train-target.txt

train:
	cd seq2seq-attn && \
		th train.lua \
			-data_file ../data/data-train.hdf5 \
			-val_data_file ../data/data-val.hdf5 \
			-savefile ../data/model

clean:
	rm -f data/train-*.txt data/*.{hdf5,dict}

.PHONY: train clean