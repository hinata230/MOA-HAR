# MOAdata Human Activity Recognition

## 0. 수행 환경
* Pytorch 2.2.2, CUDA 11.8 version에서 수행함.
  
* Pytorch 설치 (Pytorch 2.2.2, CUDA 11.8 ver.)
```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```


## 0. 사전 작업
* `LSTM/`, `Transformer/` 폴더 안에 `dataset/` 폴더를 생성한 후 사용할 데이터셋(public 데이터 혹은 수집한 데이터)을 다운로드.
  * 모든 데이터셋은 `dataset/` 폴더 안에 위치해야 함.
  ```
  cd LSTM/ (cd Transformer/)
  mkdir dataset/
  cd dataset/
  ```
  * public 데이터는 아래와 같이 다운로드.
    * UCI 데이터셋 다운로드
    ```
    wget https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
    unzip human+activity+recognition+using+smartphones.zip
    unzip UCI\ HAR\ Dataset.zip
    mv UCI\ HAR\ Dataset UCI_dataset
    ```


    * WISDM 데이터셋 다운로드 (**주의** - 다운로드 후 `WISDM_ar_v1.1_raw.txt` 파일 내 일부 line을 수정해야 함.)
    ```
    wget https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
    tar -zxvf WISDM_ar_latest.tar.gz
    mv WISDM_ar_v1.1/ WISDM_dataset
    ```

    * MotionSense 데이터셋 다운로드
    ```
    cd ../
    git init
    git remote add origin https://github.com/mmalekzadeh/motion-sense.git
    git config core.sparsecheckout true
    echo "data/A_DeviceMotion_data.zip" >> .git/info/sparse-checkout
    echo "data/data_subjects_info.csv" >> .git/info/sparse-checkout
    git pull origin master
    cd data/
    unzip A_DeviceMotion_data.zip -d ../
    mv ../A_DeviceMotion_data/ ../dataset/MotionSense_dataset
    mv data_subjects_info.csv ../dataset/MotionSense_dataset/
    ```  

  * public 데이터 링크
    * UCI : https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    * WISDM : https://www.cis.fordham.edu/wisdm/dataset.php
    * MotionSense : https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
 
* 다운로드 한 데이터셋을 .pickle로 변환시켜야 함.
  ```python
  cd Transformer/preprocess/
  python3 create_UCI.py     # UCI.train.pickle, UCI.test.pickle 생성 (public 데이터)
  python3 create_WISDM.py     # WISDM.train.pickle, WISDM.test.pickle 생성 (public 데이터)
  python3 create_MOTIONSENSE.py     # MOTIONSENSE.train.pickle, MOTIONSENSE.test.pickle 생성 (public 데이터)
  python3 create_MOA.py     # MOA.train.pickle, MOA.test.pickle 생성 (수집한 데이터) 
  ```


## 1. Softmax Classifier 및 LSTM 모델 실행 방법 
### 1. 학습하기.
* `LSTM/` 폴더로 이동 후 학습할 모델과 학습할 데이터를 입력하고 학습 모드를 지정하는 스크립트를 실행.
  
  ```
  cd LSTM
  python3 main.py --model SC --data UCI --mode train
  ```

    * --model : 사용할 모델 종류 ("SC" : Softmax Classifier, "LSTM" : LSTM)
    * --data : 사용할 데이터 종류 ("UCI", "WISDM", "MotionSense", MOA(수집한 데이터))
    * --mode : 학습/평가 모드 종류 ("train" : 학습, "eval" : 평가)

* 학습이 완료 되면 `LSTM/pretrained` 폴더가 생성되고, 폴더 안에 `model_with_params_(모델 종류)_(데이터 종류).pth` 파일이 생성됨.
  * ex) `model_with_params_SC_UCI.pth`

### 2. 평가하기.
* 평가할 모델과 데이터에 대한 `.pth` 파일이 생성되었는지 확인 필요.
* 만약 없다면 1. 학습하기. 파트에서 학습 모드 수행 후 `.pth`파일을 생성.
  
* `LSTM/` 폴더로 이동 후 평가할 모델과 평가할 데이터를 입력하고 평가 모드를 지정하는 스크립트를 실행.
  
  ```
  cd LSTM
  python3 main.py --model SC --data UCI --mode eval
  ```
* 평가가 완료되면 해당 모델에서 평가한 데이터 셋의 accuracy와 F1 Score가 출력됨.

## 2. Transformer 모델 실행 방법 
### 1. 학습하기.
* `Transformer/` 폴더로 이동 후 학습할 모델과 학습할 데이터를 입력하고 학습 모드를 지정하는 스크립트를 실행.
  
  ```
  cd Transformer
  python3 main.py --task UCI --mode train
  ```

    * --task : 사용할 데이터 종류 ("UCI", "WISDM", "MOTIONSENSE", MOA(수집한 데이터))
    * --mode : 학습/평가 모드 종류 ("train" : 학습, "eval" : 평가)

* 학습이 완료 되면 `Transformer/pretrained` 폴더가 생성되고, 폴더 안에 `model_with_params_transformer_(데이터 종류).pth` 파일이 생성됨.
  * ex) `model_with_params_transformer_UCI.pth`

### 2. 평가하기.
* 평가할 모델과 데이터에 대한 `.pth` 파일이 생성되었는지 확인 필요.
* 만약 없다면 1. 학습하기. 파트에서 학습 모드 수행 후 `.pth`파일을 생성.
  
* `Transformer/` 폴더로 이동 후 평가할 모델과 평가할 데이터를 입력하고 평가 모드를 지정하는 스크립트를 실행.
  
  ```
  cd Transformer
  python3 main.py --data task --mode eval
  ```
* 평가가 완료되면 Transformer에서 평가한 데이터 셋의 accuracy와 F1 Score가 출력됨.
