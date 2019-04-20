# Annotate based on trained model (Needs a model downloaded - here de.marmot)
java -Xmx8G -cp marmot.jar marmot.morph.cmd.Annotator --model-file da.marmot --test-file form-index=1,data/da-dev.conllu --pred-file da-dev.out.txt

time
real    0m14.944s
user    0m0.000s
sys     0m0.031s

# Train new model
java -Xmx8G -cp marmot.jar marmot.morph.cmd.Trainer -train-file form-index=1,tag-index=3,data/da-train.conllu -tag-morph false -model-file da.marmot

time
real    29m17.651s
user    0m0.000s
sys     0m0.015s