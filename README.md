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

conda create -n mlframework36 python=3.6 anaconda
source activate mlframework36
conda install anaconda-client  #needed for binstart yaml
conda install -n mlframework36 environment.yml

chmod +x run.sh; ./run.sh


```

[abhishek repo](https://github.com/abhishekkrthakur/mlframework)
[video repo](https://www.youtube.com/watch?v=ArygUBY0QXw&feature=youtu.be)
