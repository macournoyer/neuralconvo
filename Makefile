DATA=$(shell pwd)/data

$(DATA)/data-train.hdf5: $(DATA)/train-source.txt
	cd seq2seq-attn && python preprocess.py \
		--srcfile $(DATA)/train-source.txt \
		--targetfile $(DATA)/train-target.txt \
		--srcvalfile $(DATA)/validation-source.txt \
		--targetvalfile $(DATA)/validation-target.txt \
		--outputfile $(DATA)/data

$(DATA)/train-source.txt:
	th prepreprocess.lua --srcfile $(DATA)/train-source.txt --targetfile $(DATA)/train-target.txt

train:
	cd seq2seq-attn && \
		th train.lua \
			-data_file $(DATA)/data-train.hdf5 \
			-val_data_file $(DATA)/data-val.hdf5 \
			-savefile $(DATA)/model

clean:
	rm -f $(DATA)/train-*.txt $(DATA)/*.{hdf5,dict}

.PHONY: clean