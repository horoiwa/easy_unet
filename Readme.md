### 二値化ツール
---

#### 使い方
※入力はグレスケです.3チャネル画像の場合はグレスケ化されます.
  
【training】
1. samples/train/imageにグレスケの元画像を収納（JPEGのみ）
2. samples/train/maskにグレスケの教師画像を収納(JPEGのみ)
  
`python train.py -n_sample X -steps Y -epochs Z`  


>コマンドライン引数  
>n_sample: 元画像から256×256の画像の復元再抽出を何回行うか  
steps: steps per epoch, デフォルト2000   
epochs: エポック数,　デフォルト5  
    
  
【test】
1. samples/test/image にグレスケのテスト画像を収納(JPEGのみ)
  
`python test.py`  
  
result/に結果が出力される

  
【初期化】  
`python cleanup.py`  


#### 注意
拡張子は.jpg　（.jpeg .JPGなどは不可）
  
テスト画像の縦横サイズはそれぞれ256の倍数であることが望ましい  
そうでない場合はリサイズされる
  

---