let imgElement = document.getElementById('imageSrc');
let inputElement = document.getElementById('fileInput');
inputElement.addEventListener('change', (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);

function get_plate_img(){
    let mat = cv.imread(imgElement);
    // debug
    // let dst_d=binarize_with_color_info(mat);
    let dst_d=resize_image(mat,2);
    cv.imshow('canvasOutput_debug', dst_d);
    // processing img
    let dst=processing(mat);
    cv.imshow('canvasOutput', dst);
    mat.delete();
    dst.delete();
}

// 画像のリサイズ関数
function resize_image(src,ratio){
    let dst = new cv.Mat();
    // get original image size
    let resized_rows=Math.round(src.rows*ratio);
    let resized_cols=Math.round(src.cols*ratio);
    if (ratio>=1){
        cv.pyrUp(src, dst, new cv.Size(resized_cols, resized_rows), cv.BORDER_DEFAULT);
    }else{
        cv.pyrDown(src, dst, new cv.Size(resized_cols, resized_rows), cv.BORDER_DEFAULT);
    }
    return dst;
}

// RGB範囲内か否かで二値化する処理
function binarize_with_color_info(src){
    let dst = new cv.Mat();
    let low = new cv.Mat(src.rows, src.cols, src.type(), [130, 140, 140, 0]);
    let high = new cv.Mat(src.rows, src.cols, src.type(), [255, 255, 255, 255]);
    cv.inRange(src, low, high, dst);
        return dst;
}


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
}

function processing(src){
    let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);

    //// binarize
    src= binarize_with_color_info(src);

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
        cv.approxPolyDP(cnt, tmp, cnt_length*0.01, false);
        let area = cv.contourArea(tmp, false);
        let polygonnum = tmp.data32S.length;
        if (polygonnum<14 && polygonnum>8 && area>400){
            poly.push_back(tmp);
        }
        cnt.delete();
        tmp.delete();
    }
    // draw contours with random Scalar
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

function onOpenCvReady() {
  document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
}
