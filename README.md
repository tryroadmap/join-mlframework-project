# setup

#### input data from Kaggle
input `kaggle competitions download -c titanic`. [Install Kaggle API.](https://github.com/Kaggle/kaggle-api)
#### input data wget
```
git clone https://github.com/lotusxai/mlframework.git

s3_address=https://phillytalent.s3.amazonaws.com/projects/titanic/data/titanic.zip
wget $s3_address

#atom mlframework
mv *.zip ./mlframework/input/
cd mlframework;

conda update conda
conda upgrade conda
conda --version #4.7 above

conda create -n mlframework36 python=3.6 anaconda

source activate mlframework36
conda env update --file environment.yml
conda info --envs


chmod +x run.sh; ./run.sh
source deactivate
#conda remove -n mlframework36 -all
```

[abhishek repo](https://github.com/abhishekkrthakur/mlframework)
[video repo](https://www.youtube.com/watch?v=ArygUBY0QXw&feature=youtu.be)

# Execution

Prior to execution, desired models should be added to the `src/dispatcher.py` dictionary structure. Code is executed using `run.sh` as the point of entry, with the model key included as an argument. As an example:

```
./run.sh randomforest
```

This will perform and store training on the specified random forest model, automatically report validation (training) accuracy, and produce a csv of predictions on the test data set, formatted to Kaggle standards. When making nontrivial adjustments to the model, you will be editing the following scripts:

- `src/train.py`: Cleans and splits data, performs feature engineering, validates and stores model (pipeline)
- `src/test.py`: Load training model, apply to (cleaned) testing data, and store result in `models/{model_name}.csv`
- `src/dispatcher.py`: A dictionary containing the models recognized by the framework
- `src/params.json`: Contains basic information about data that cannot be automatically extracted
  - `target`: The name of the target variable to be predicted
  - `id`: The name of the row identifier, used for producing the final csv
  - `dropFeatures`: A list of feature names that you would not like to use for training purposes
  - `categoricals`: A list of feature names that you would like to be treated as categorical variables (all else treated numerically by default)


At this stage in development, **it is important to be aware that custom changes made to data in training (i.e. changes not added directly to the pipeline) must be mirrored in the data-testing routine!**
