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
