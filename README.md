


# Set up 

```
npm install -g http-server
```

# How to start test
## サーバー立ち上げ
```
http-server
```

## IPの変更
main.js のモデルロードのIPを実態に合わせて修正

## メインページ
main.htmlを開いてください
```
http://IPアドレス:ポート番号/main.html
```

# モデルの変更
モデルの変更が必要な場合には、tfjs/modelのjsonを変更します。
その時、main.jsのINPUT_SIZEやFEATURE_SIZEも必要に応じて変更


# install opencv

Officially you can build from source
https://docs.opencv.org/3.3.1/d4/da1/tutorial_js_setup.html

you can also download from tutorial instead.
'''
curl https://docs.opencv.org/4.3.0/opencv.js -o opencv.js
'''


### ref

See open cv [tutorial](https://docs.opencv.org/3.4/d2/df0/tutorial_js_table_of_contents_imgproc.html)"# js-licence-plate-recognition" 