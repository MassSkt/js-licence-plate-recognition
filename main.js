let imgElement = document.getElementById('imageSrc');
let inputElement = document.getElementById('fileInput');
inputElement.addEventListener('change', (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);

// https://js.tensorflow.org/api/1.2.6/
const DEFINED_HEIGHT=700;

var model=0;

async function run(){
    // load model
    const path = "http://192.168.100.105:8080/tfjs/model.json";
    // const model = await tf.loadLayersModel(path);
    model = await tf.loadLayersModel(path);
    // predict
    // const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    // y_pred = await model.predict(xs);
    // y_pred.print();

    // convert to array
    // const values = await y_pred.data();
    // const arr = await Array.from(values);
    // console.log(arr);
  }
run();
function get_plate_img_tf(){
    let img=document.getElementById('imageSrc');
    document.getElementById('result_text').innerHTML = 'pred start';
    document.getElementById('result_text').innerHTML = 1;
    predict(imgElement);

}

async function predict(imgElement) {
    // status('Predicting...');
  
    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    // const startTime1 = performance.now();
    // The second start time excludes the extraction and preprocessing and
    // includes only the predict() call.
    // let startTime2;
    const logits = tf.tidy(() => {
        // tf.browser.fromPixels() returns a Tensor from an image element.
        const tfimg = tf.browser.fromPixels(imgElement).toFloat();

        const pred=get_plate(tfimg);


        document.getElementById('result_text').innerHTML = pred;
  
    //   const offset = tf.scalar(127.5);
    //   // Normalize the image from [0, 255] to [-1, 1].
    //   const normalized = img.sub(offset).div(offset);
  
    //   // Reshape to a single-element batch so we can pass it to predict.
    //   const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
  
    //   startTime2 = performance.now();
    //   // Make a prediction through mobilenet.
    //   return mobilenet.predict(batched);
    });
  // Convert logits to probabilities and class names.
//   const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
//   const totalTime1 = performance.now() - startTime1;
//   const totalTime2 = performance.now() - startTime2;
//   status(`Done in ${Math.floor(totalTime1)} ms ` +
//       `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

//   // Show the classes in the DOM.
//   showResults(imgElement, classes);
}
function get_plate(tfArray, Dmax=608, Dmin=256){
    // normalize 255
    const normalized_tfArray=preprocess_image(tfArray);
    const ratio=Math.max(...normalized_tfArray.shape.slice(0, 2))/Math.min(...normalized_tfArray.shape.slice(0, 2));
    const side = Math.round(ratio*Dmin);
    const bound_dim = Math.min(side, Dmax);
    console.log(bound_dim);
    const min_dim_img = Math.min(...normalized_tfArray.shape.slice(0, 2))
    console.log(min_dim_img);
    const factor = bound_dim / min_dim_img;
    console.log(normalized_tfArray.shape);
    console.log(factor);
    const w = Math.round(normalized_tfArray.shape[1] * factor);
    const h = Math.round(normalized_tfArray.shape[0] * factor);
    console.log(h,w);

    const resized_tfArray = tf.image.resizeBilinear (normalized_tfArray, [h,w], false);
    // const resized_tfArray=tf.ones([2,2,3]);
    console.log(resized_tfArray.shape);
    const reshaped_rs_tfArray = resized_tfArray.reshape([1, resized_tfArray.shape[0],resized_tfArray.shape[1],resized_tfArray.shape[2]]);
    console.log(reshaped_rs_tfArray.shape);
    

    const Yr = model.predict(reshaped_rs_tfArray);
    console.log(Yr);
    console.log(Yr.shape);
    // Yr.print();
    // load model
    // const path = "http://localhost:8080/tfjs/model.json";
    // const model = await tf.loadLayersModel(path);

    // vehicle = preprocess_image(src);
    // ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    // side = int(ratio * Dmin)
    // bound_dim = min(side, Dmax)
    // detect_lp(model, I, max_dim, lp_threshold):
    // min_dim_img = min(I.shape[:2])
    // factor = float(max_dim) / min_dim_img # w>h w / h * Dmin / h
    // w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist() # w * w / h * Dmin / h, h * w / h * Dmin / h
    // Iresized = cv2.resize(I, (w, h))
    // T = Iresized.copy()
    // T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    // print("detect lp: I shape{},factor{},w{},h{},Iresised shape{},T shape{}".format(I.shape,factor,w,h,Iresized.shape,T.shape))
    // Yr = model.predict(T)
    // Yr = np.squeeze(Yr)

    // _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    // return LpImg, cor
    return Yr.shape;
}

function preprocess_image(tfArray){
    const normalized_tfArray=tfArray.div(tf.scalar(255));
    return normalized_tfArray;
};
// tf.loadModel('http://localhost:8080/tfjs/model.json').then(handleModel).catch(handleError);
// function handleModel(model) {
//     // 正常に読み込まれた時の処理
//     console.log("model load success");
//     // 必要なら入出力shapeを保存
//     height = model.inputs[0].shape[1];
//     width = model.inputs[0].shape[2];
//     // modelの操作...
//     console.log(height,width);
// }
// function handleError(error) {
//     console.log("model load error");
//     // エラー処理
// }

function get_plate_img(){
    let mat = cv.imread(imgElement);
    // debug
    // let dst_d=binarize_with_color_info(mat);
    let resize_ratio=DEFINED_HEIGHT/mat.rows;
    let dst_d=resize_image(mat,resize_ratio);
    dst_d=sharp_edge(dst_d);
    dst_d=binarize_with_color_info(dst_d);
    dst_d=erode_image(dst_d)
    cv.imshow('canvasOutput_debug', dst_d);
    // processing img
    let dst=processing(mat);
    cv.imshow('canvasOutput', dst);
    mat.delete();
    dst.delete();
};

// 画像のリサイズ関数
function resize_image(src,ratio){
    let dst = new cv.Mat();
    // get original image size
    cv.resize(src, dst, new cv.Size(0,0),ratio,ratio, cv.INTER_NEAREST );
    return dst;
};

// RGB範囲内か否かで二値化する処理
function binarize_with_color_info(src){
    let dst = new cv.Mat();
    let low = new cv.Mat(src.rows, src.cols, src.type(), [120, 120, 120, 0]);
    let high = new cv.Mat(src.rows, src.cols, src.type(), [255, 255, 255, 255]);
    cv.inRange(src, low, high, dst);
        return dst;
};


// RGB条件式に従って二値化する処理（ナイーブ）
function binarize_with_color_info_(src){
    let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
    if (src.isContinuous()) {
        for (let row = 0; row < src.rows; ++row) {
            for (let col = 0; col < src.cols; ++col) {
                let R = src.data[row * src.cols * src.channels() + col * src.channels()];
                let G = src.data[row * src.cols * src.channels() + col * src.channels() + 1];
                let B = src.data[row * src.cols * src.channels() + col * src.channels() + 2];
                let A = src.data[row * src.cols * src.channels() + col * src.channels() + 3];
                if ((R>130) && (G>140) && (B>140)){
                    // dst.data[row * src.cols + col]=255
                    dst.ucharPtr(row,col)[0]=255
                }else{
                    // dst.data[row * src.cols + col]=0
                    dst.ucharPtr(row,col)[0]=0
                }
            }
        }
    }
    return dst;
};

function sharp_edge(src){
    const k = -1.0;
    let dst = new cv.Mat();
    // let M = cv.Mat.eye(3, 3, cv.CV_32FC1);
    // let M = cv.matFromArray(3, 3, cv.CV_32FC1, [k, k, k, k, 9.0, k, k, k, k]);
    let M = cv.matFromArray(3, 3, cv.CV_32FC1, [ 0.0, k, 0.0, k, 5.0, k, 0.0, k, 0.0]);
    let anchor = new cv.Point(-1, -1);
    // You can try more different parameters
    cv.filter2D(src, dst, cv.CV_8U, M, anchor, 0, cv.BORDER_DEFAULT);
    return dst
};

function erode_image(src){
    let M = cv.Mat.ones(2, 2, cv.CV_8U);
    let dst = new cv.Mat();
    let anchor = new cv.Point(-1, -1);
    cv.erode(src, dst, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    return dst;

};

function processing(src){

    //  resize (unify size)
    let resize_ratio=DEFINED_HEIGHT/src.rows;
    src=resize_image(src,resize_ratio);

    // shap edge
    src=sharp_edge(src);

    //// binarize
    src= binarize_with_color_info(src);
    
    // erode
    src =erode_image(src)

    let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);

    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    let poly = new cv.MatVector();
    cv.findContours(src, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    // approximates each contour to polygon
    for (let i = 0; i < contours.size(); ++i) {
        let tmp = new cv.Mat();
        let cnt = contours.get(i);
        // You can try more different parameters
        let cnt_length = cv.arcLength(cnt, true);
        console.log(cnt_length)
        // cv.approxPolyDP(cnt, tmp, 3, false);
        cv.approxPolyDP(cnt, tmp, cnt_length*0.05, false);
        let area = cv.contourArea(tmp, false);
        let polygonnum = tmp.data32S.length;
        // if (true){
        if (polygonnum<14 && polygonnum>8 && area>400){
        // if (polygonnum<14 && polygonnum>8 && area>400){
            poly.push_back(tmp);
        }
        cnt.delete();
        tmp.delete();
    }
    // draw contours with random Scalar
    // for (let i = 0; i < contours.size(); ++i) {
    for (let i = 0; i < poly.size(); ++i) {
        let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                                  Math.round(Math.random() * 255));
        cv.drawContours(dst, poly, i, color, 1, 8, hierarchy, 0);
    }
    return dst;
};

function binarize(src){
    var dst = new cv.Mat();
    cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(dst, dst, 100, 200, cv.THRESH_BINARY);
    // cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    // cv.medianBlur(src, src, 5);
    // cv.adaptiveThreshold(src, dst, 200, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 2);
    return dst    
}

//  Open CV 読み込み完了
function onOpenCvReady() {
  document.getElementById('status_cv').innerHTML = 'OpenCV.js is ready.';
}

// tensorflow js 読み込み完了
function onTfJsReady() {
    document.getElementById('status_tf').innerHTML = 'Tf.js is ready.';
  }