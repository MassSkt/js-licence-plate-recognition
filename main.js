// Canvas 要素の読み込み
const canvas = document.getElementById("canvasInput")
const canvas_resized = document.getElementById("canvasInputResized")
const canvas_focused = document.getElementById("canvasInputFocused")
const canvas_focused_resized = document.getElementById("canvasInputFocusedResized")
document.querySelector('input[type="file"]').onchange = function() {
    let img = this.files[0]
    let reader = new FileReader()
    reader.readAsDataURL(img)
    reader.onload = function() {
        drawOrgImage(reader.result);
    }
}

// model の読み込み　https://js.tensorflow.org/api/1.2.6/
var model=0;
async function model_load(){
    // load model IP は実態に合わせて修正してください
    // const path = "http://192.168.100.105:8080/tfjs/model.json";
    const path = "http://192.168.1.5:8080/tfjs/model.json";
    model = await tf.loadLayersModel(path);
  }
model_load();

// 元画像の描画
function drawOrgImage(url) {
    let ctx = canvas.getContext('2d')
    let image = new Image()
    image.src = url
    image.onload = () => {
        canvas.width = image.width
        canvas.height = image.height
        ctx.drawImage(image, 0, 0)
    }
}

// ナンバープレートの頂点を取得する関数（デバックで描画）
function get_plate_area_points(){
    //モデルのインプット、アウトプットサイズ、モデルに応じて変更
    const INPUT_SIZE=256;
    const OUT_FEATURE_SIZE=14;

    // canvasから元画像の取得 open cv
    let startTime = performance.now(); // 開始時間
    let src_org=cv.imread(canvas);
    let width_org=src_org.cols;
    let height_org=src_org.rows;
    let endTime = performance.now(); // 終了時間
    console.log("read imge time")
    console.log(endTime - startTime); 

    // resize された画像を描画 open cv
    startTime = performance.now(); // 開始時間
    let resized_mat = resize_image_(src_org,INPUT_SIZE,INPUT_SIZE);
    cv.imshow('canvasInputResized', resized_mat);
    endTime = performance.now(); // 終了時間
    console.log("resize imge time")
    console.log(endTime - startTime); 

    //  キャンバスから取得し、TFによる予測（一回キャンバスに落とさないとTFで取得できないという前提で書いているが、正しいか不明）
    startTime = performance.now(); // 開始時間
    let img_resized=document.getElementById('canvasInputResized');
    // document.getElementById('result_text').innerHTML = 'pred start';
    const tfimg = tf.browser.fromPixels(img_resized).toFloat();
    let tf_pred=get_plate(tfimg,0.5);//tf array
    let pred_arr=tf_pred.arraySync();
    const pred_arr_ = [].concat(...pred_arr); 
    // console.log(pred_arr_);
    var pred_arr_255=[];
    for (let i = 0; i < pred_arr_.length; i++) {
        pred_arr_255.push(Math.round(pred_arr_[i]*255));
    }
    endTime = performance.now(); // 終了時間
    console.log("TF predict time")
    console.log(endTime - startTime); 

    // open cv でナンバープレート反応出力領域を抽出
    startTime = performance.now(); // 開始時間
    let pred_mat = cv.matFromArray(OUT_FEATURE_SIZE, OUT_FEATURE_SIZE, cv.CV_8UC1, pred_arr_255);
    pred_mat = resize_image_by_ratio(pred_mat,INPUT_SIZE/OUT_FEATURE_SIZE);
    let plate_rect=get_rect(pred_mat);// {x,y,w,h}
    let plate_rect_margin = get_margin_rect(plate_rect,10,INPUT_SIZE,INPUT_SIZE);
    // console.log(plate_rect_margin);
    endTime = performance.now(); // 終了時間
    console.log("extract reagion with open cv time")
    console.log(endTime - startTime); 

    // 元画像での候補領域矩形座標
    startTime = performance.now(); // 開始時間
    let plate_rect_candidate =  get_margin_rect(plate_rect,20,INPUT_SIZE,INPUT_SIZE);
    let plate_rect_org= convert_rect(plate_rect_candidate,width_org/INPUT_SIZE,height_org/INPUT_SIZE);//元座標
    let src_org_focused=src_org.roi(plate_rect_org);
    let src_org_focused_resized=resize_image_(src_org_focused,INPUT_SIZE,INPUT_SIZE);
    cv.imshow('canvasInputFocused', src_org_focused);
    cv.imshow('canvasInputFocusedResized', src_org_focused_resized);
    endTime = performance.now(); // 終了時間
    console.log("extracted region draw image time")
    console.log(endTime - startTime); 

    //候補領域画像で再検査
    //  TFによる予測
    startTime = performance.now(); // 開始時間
    let img_focus_resized=document.getElementById('canvasInputFocusedResized');
    const tfimgf = tf.browser.fromPixels(img_focus_resized).toFloat();
    let tf_predf=get_plate(tfimgf,0.6);//tf array
    let predf_arr=tf_predf.arraySync();
    const predf_arr_ = [].concat(...predf_arr); 
    // console.log(predf_arr_);
    var predf_arr_255=[];
    for (let i = 0; i < predf_arr_.length; i++) {
        predf_arr_255.push(Math.round(predf_arr_[i]*255));
    }
    endTime = performance.now(); // 終了時間
    console.log("TF predict time")
    console.log(endTime - startTime); 

    // open cv
    startTime = performance.now(); // 開始時間
    let predf_mat = cv.matFromArray(14, 14, cv.CV_8UC1, predf_arr_255);
    predf_mat = resize_image_by_ratio(predf_mat,256/14);
    cv.imshow('canvasOutput_debug5', predf_mat);
    endTime = performance.now(); // 終了時間
    console.log("draw extracted image predict time")
    console.log(endTime - startTime); 

    // ret_xs=mul_num_to_arr(ret_xs,plate_rect_org['width']/INPUT_SIZE);

    // プレート領域取得
    startTime = performance.now(); // 開始時間
    let plate_rect_focus=get_rect(predf_mat);
    let plate_rect_focus_margin =  get_margin_rect(plate_rect_focus,10,INPUT_SIZE,INPUT_SIZE);
    let plate_focus_margin_mat = src_org_focused_resized.roi(plate_rect_focus_margin);
    cv.imshow('canvasOutput_debug55', plate_focus_margin_mat); //debug
    // console.log(plate_rect_focus)
    let img_focus_resized_mat=cv.imread(canvas_focused_resized)
    let plate_mat_focus = img_focus_resized_mat.roi(plate_rect_focus);
    cv.imshow('canvasOutput_debug6', plate_mat_focus); //debug
    endTime = performance.now(); // 終了時間
    console.log("plateregion extract time")
    console.log(endTime - startTime); 

    startTime = performance.now(); // 開始時間
    cv.cvtColor(plate_mat_focus, plate_mat_focus, cv.COLOR_RGB2HSV, 0);
    cv.cvtColor(plate_focus_margin_mat, plate_focus_margin_mat, cv.COLOR_RGB2HSV, 0);
    let srcVec = new cv.MatVector();
    let dstVec = new cv.MatVector();
    srcVec.push_back(plate_mat_focus); dstVec.push_back(plate_focus_margin_mat);
    let backproj = new cv.Mat();
    let none = new cv.Mat();
    let mask = new cv.Mat();
    let hist = new cv.Mat();
    let channels = [0,1,2];
    let histSize = [11,11,11];
    let ranges = [0, 256,0,256,0,256];
    let accumulate = false;
    cv.calcHist(srcVec, channels, mask, hist, histSize, ranges, accumulate);
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX, -1, none);
    cv.calcBackProject(dstVec, channels, hist, backproj, ranges, 1);
    cv.imshow('canvasOutput_debug7', backproj);
    let backproj_thresh=binarize_with_color_info(backproj,[100,100,100,0],[255,255,255,255]);
    backproj_thresh=erode_image(backproj_thresh,3);
    backproj_thresh=dilate_image(backproj_thresh,20);
    backproj_thresh=erode_image(backproj_thresh,10);
    cv.imshow('canvasOutput_debug8', backproj_thresh);
    endTime = performance.now(); // 終了時間
    console.log("backproj time")
    console.log(endTime - startTime); 

    startTime = performance.now(); // 開始時間
    let {return_xs,return_ys}=get_inplate_cnt(backproj_thresh);
    // in plate rect margin , convert coordinate
    // ret_xs=mul_num_to_arr(ret_xs,plate_rect_focus_margin['width']/INPUT_SIZE);
    // return_ys=mul_num_to_arr(return_ys,plate_rect_focus_margin['height']/INPUT_SIZE);
    return_xs=add_num_to_arr(return_xs,plate_rect_focus_margin['x']);
    return_ys=add_num_to_arr(return_ys,plate_rect_focus_margin['y']);
    return_xs=mul_num_to_arr(return_xs,plate_rect_org['width']/INPUT_SIZE);
    return_ys=mul_num_to_arr(return_ys,plate_rect_org['height']/INPUT_SIZE);
    return_xs=add_num_to_arr(return_xs,plate_rect_org['x']);
    return_ys=add_num_to_arr(return_ys,plate_rect_org['y']);
    endTime = performance.now(); // 終了時間
    console.log("fine extraction time")
    console.log(endTime - startTime); 

    // debug draw
    startTime = performance.now(); // 開始時間
    var canvas_out = document.getElementById('canvasOutput_debug4');
    var c = canvas_out.getContext('2d');
    let image_out = new Image()
    image_out.src = canvas.toDataURL("image/png");
    image_out.onload = () => {
        canvas_out.width = image_out.width;
        canvas_out.height = image_out.height;
        c.drawImage(image_out, 0, 0);
        // 三角形　１つ目
        c.strokeStyle = 'red';  // 線の色
        // パスの開始
        c.beginPath();
        for (let i = 0; i < return_xs.length; ++i) {
            if (i==0){
                c.moveTo(return_xs[i],return_ys[i]);
            }else{
                c.lineTo(return_xs[i],return_ys[i]);
            }
        }
        c.closePath();
        // 描画
        c.stroke();
    }
    endTime = performance.now(); // 終了時間
    console.log("draw NP rect time")
    console.log(endTime - startTime); 

    return {return_xs,return_ys}

    
}


function add_num_to_arr(arr,num){
    let ret_arr=[];
    for (let i = 0; i < arr.length; ++i) {
        ret_arr.push(arr[i]+num);
    }
    return ret_arr;
}
function mul_num_to_arr(arr,num){
    let ret_arr=[];
    for (let i = 0; i < arr.length; ++i) {
        ret_arr.push(Math.round(arr[i]*num));
    }
    return ret_arr;
}

// rect矩形領域にマージンを加えて出力
function get_margin_rect(rect,margin,xlim,ylim){
    let rect_margin={}
    rect_margin['x']=Math.max(0,rect['x']-margin);
    rect_margin['y']=Math.max(0,rect['y']-margin);
    rect_margin['width']=Math.min(xlim,rect['width']+rect['x']+2*margin)-rect['x'];
    rect_margin['height']=Math.min(ylim,rect['height']+rect['y']+2*margin)-rect['y'];
    return rect_margin;
}

function convert_rect(rect,width_ratio,height_ratio){
    let ret_rect={};//元座標
    ret_rect['x']=Math.round(rect['x']*width_ratio);
    ret_rect['y']=Math.round(rect['y']*height_ratio);
    ret_rect['width']=Math.round(rect['width']*width_ratio);
    ret_rect['height']=Math.round(rect['height']*height_ratio);
    return ret_rect;

}


// 二値画像からplate領域座標（4点）を取得する関数
function get_inplate_cnt(src){

    // コンターの取得
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    let poly = new cv.MatVector();
    cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    // 　複数のコンター（１フラグが立った領域）について最も中心のポリゴンを取得
    let center_score=10000;
    let ret_xs=[];
    let ret_ys=[];
    // let center={x:0,y:0};
    let min_rect_vertices=[];
    for (let i = 0; i < contours.size(); ++i) {
        let tmp = new cv.Mat();
        let cnt = contours.get(i);
        // You can try more different parameters
        let cnt_length = cv.arcLength(cnt, true);
        // console.log(cnt_length)
        // cv.approxPolyDP(cnt, tmp, 3, false);
        cv.approxPolyDP(cnt, tmp, cnt_length*0.06, false);
        let area = cv.contourArea(tmp, false);
        let polygonnum = tmp.data32S.length;
        if (true){
        // if (polygonnum<24 && polygonnum>4 && area>80){
        // if (polygonnum<14 && polygonnum>8 && area>400){
            // console.log(tmp.data32S);
            // console.log(tmp);
            let temp_x=[];
            let temp_y=[];
            for (let k = 0; k < tmp.data32S.length/2; ++k){
                temp_x.push(tmp.data32S[k*2]);
                temp_y.push(tmp.data32S[k*2+1]);
            }
            // console.log(average_arr_int(temp_x));
            // console.log(average_arr_int(temp_y));
            let center_x=average_arr_int(temp_x);
            let center_y=average_arr_int(temp_y);
            let temp_center_score=Math.abs(src.cols/2-center_x)+Math.abs(src.rows/2-center_y);
            // console.log(temp_center_score);
            if (temp_center_score<center_score){
                center_score=temp_center_score;
                ret_xs=temp_x;
                ret_ys=temp_y;
                // center = cv.minEnclosingCircle(cnt).center;//{x:,y:}
                let rotatedRect = cv.minAreaRect(cnt);
                min_rect_vertices = cv.RotatedRect.points(rotatedRect);//vertices [{x:,y:},{x:,y:}]
            }

            poly.push_back(tmp);
        }
        cnt.delete();
        tmp.delete();
    }

    //頂点の4点に収束
    let {return_xs,return_ys}=get_4points_from_polygon(ret_xs,ret_ys,min_rect_vertices);

    // DEBUG draw contours with random Scalar Debug
    let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);
    for (let i = 0; i < poly.size(); ++i) {
        let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                                  Math.round(Math.random() * 255));
        cv.drawContours(dst, poly, i, color, 1, 8, hierarchy, 0);
    }
    cv.imshow('canvasOutput_debug3', dst);

    console.log({return_xs,return_ys});
    return {return_xs,return_ys};
};

function get_4points_from_polygon(xs,ys,min_rect_vertices){
    // 中心点の計算
    // let center_x=average_arr_int(xs);
    // let center_y=average_arr_int(ys);
    // let center_x=center.x;
    // let center_y=center.y;
    let min_rect_xs=[];
    let min_rect_ys=[];
    for (let i=0; i<4; ++i){
        min_rect_xs.push(min_rect_vertices[i].x);
        min_rect_ys.push(min_rect_vertices[i].y);
    }
    let center_x=average_arr_int(min_rect_xs);
    let center_y=average_arr_int(min_rect_ys);

    let top_left_idx=0;
    let top_right_idx=0;
    let bottom_left_idx=0;
    let bottom_right_idx=0;
    
    let top_left_score=0;
    let top_right_score=0;
    let bottom_left_score=0;
    let bottom_right_score=0;

    let top_left_score_temp=0;
    let top_right_score_temp=0;
    let bottom_left_score_temp=0;
    let bottom_right_score_temp=0;

    console.log(xs);
    console.log(ys);
    console.log(center_x);
    console.log(center_y);
    // 中心から各角（遠い点）を算出
    for (let i = 0; i < xs.length; ++i){
        top_left_score_temp=(center_x-xs[i])+(center_y-ys[i]);
        top_right_score_temp=(xs[i]-center_x)+(center_y-ys[i]);
        bottom_left_score_temp=(center_x-xs[i])+(ys[i]-center_y);
        bottom_right_score_temp=(xs[i]-center_x)+(ys[i]-center_y);
        if (top_left_score_temp > top_left_score){
            top_left_idx=i;
            top_left_score=top_left_score_temp;
        }
        if (top_right_score_temp > top_right_score){
            top_right_idx=i;
            top_right_score=top_right_score_temp;
        }
        if (bottom_left_score_temp > bottom_left_score){
            bottom_left_idx=i;
            bottom_left_score=bottom_left_score_temp;
        }
        if (bottom_right_score_temp > bottom_right_score){
            bottom_right_idx=i;
            bottom_right_score=bottom_right_score_temp;
        }
    }

    let return_xs=[];
    let return_ys=[];

    return_xs.push(xs[top_left_idx]);
    return_xs.push(xs[top_right_idx]);
    return_xs.push(xs[bottom_right_idx]);
    return_xs.push(xs[bottom_left_idx]);

    return_ys.push(ys[top_left_idx]);
    return_ys.push(ys[top_right_idx]);
    return_ys.push(ys[bottom_right_idx]);
    return_ys.push(ys[bottom_left_idx]);

    // return_xs=min_rect_xs;
    // return_ys=min_rect_ys;

    return {return_xs,return_ys}

}


function get_plate_rgb(src){
    let src_hsv = new cv.Mat();
    cv.cvtColor(src, src_hsv, cv.COLOR_RGB2HSV, 0);//get hsv image too
    console.log(src_hsv.channels);
    let white_plate_R=[];
    let white_plate_G=[];
    let white_plate_B=[];
    let yellow_plate_R=[];
    let yellow_plate_G=[];
    let yellow_plate_B=[];
    let black_plate_R=[];
    let black_plate_G=[];
    let black_plate_B=[];
    
    for (let row = 0; row < src.rows; ++row) {
        for (let col = 0; col < src.cols; ++col) {
            let R = src.data[row * src.cols * src.channels() + col * src.channels()];
            let G = src.data[row * src.cols * src.channels() + col * src.channels() + 1];
            let B = src.data[row * src.cols * src.channels() + col * src.channels() + 2];
            let A = src.data[row * src.cols * src.channels() + col * src.channels() + 3];
            let H = src_hsv.data[row * src_hsv.cols * src_hsv.channels() + col * src_hsv.channels()]
            // console.log(H);
            // white plate cond
            if ((R>120) && (G>130) && (B>130)){
                white_plate_R.push(R);
                white_plate_G.push(G);
                white_plate_B.push(B);
            //yellow plate cond
            }else if((30*255/360<H && (H<60*255/360))){
                yellow_plate_R.push(R);
                yellow_plate_G.push(G);
                yellow_plate_B.push(B);
            // black plate cond
            }else if((R<35) && (G<35) && (B<35)){
                black_plate_R.push(R);
                black_plate_G.push(G);
                black_plate_B.push(B);
            }
        }
    }
    console.log(white_plate_R);
    console.log(yellow_plate_R);
    console.log(black_plate_R);
    let total_pixels=white_plate_R.length+yellow_plate_R.length+black_plate_R.length;
    //　ある程度 30% シロ領域あり、シロ＞黄色ならば
    if ((white_plate_R.length/total_pixels)>0.4 && (white_plate_R.length > yellow_plate_R.length)){
        let ret_R=average_arr_int(white_plate_R);
        let ret_G=average_arr_int(white_plate_G);
        let ret_B=average_arr_int(white_plate_B);
        console.log({ret_R,ret_G,ret_B});
        return({ret_R,ret_G,ret_B});
    }else if((yellow_plate_R.length/total_pixels)>0.08){
        let ret_R=average_arr_int(yellow_plate_R);
        let ret_G=average_arr_int(yellow_plate_G);
        let ret_B=average_arr_int(yellow_plate_B);
        console.log({ret_R,ret_G,ret_B});
        return({ret_R,ret_G,ret_B});
    }else{
        let ret_R=average_arr_int(black_plate_R);
        let ret_G=average_arr_int(black_plate_G);
        let ret_B=average_arr_int(black_plate_B);
        console.log({ret_R,ret_G,ret_B});
        return({ret_R,ret_G,ret_B});
    }
    // return dst;
};

function average_arr_int(arr){
    let sum=0;
    for (let i = 0; i < arr.length; ++i) {
        sum += arr[i];
    }
    let average = sum/arr.length;
    average = Math.round(average);
    return average;
}

function get_rect(mat){
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(mat, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    // let cnt = contours.get(0);
    // let rect = cv.boundingRect(cnt);
    let rect=0;
    let area=0;
    for (let i = 0; i < contours.size(); ++i) {
        let cnt = contours.get(i);
        let area_temp = cv.contourArea(cnt, false);

        if (i==0){
            rect = cv.boundingRect(cnt);
            area = area_temp;
        }else{
            let rect_temp =  cv.boundingRect(cnt);
            // if (rect_temp['y']>rect['y']){
            //     rect=rect_temp;
            // }
            if (area_temp>area){
                rect=rect_temp;
            }
        }
        cnt.delete();
    }
    // console.log(rect);
    return rect;
    
}


async function predict(imgElement) {
    // status('Predicting...');
  
    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    const logits = tf.tidy(() => {
        // tf.browser.fromPixels() returns a Tensor from an image element.
        const tfimg = tf.browser.fromPixels(imgElement).toFloat();

        const pred=get_plate(tfimg,0.5);


        document.getElementById('result_text').innerHTML = pred;
        return pred;
  

    });
}
function get_plate(tfArray, threshold){
    // normalize 255
    const INPUT_SIZE=256
    const thresh_arr=tf.scalar(threshold);
    const resized_tfArray = tf.image.resizeNearestNeighbor (tfArray, [INPUT_SIZE,INPUT_SIZE], false);
    const normalized_tfArray=preprocess_image(resized_tfArray);

    // console.log(resized_tfArray.shape);
    const reshaped_rs_tfArray = normalized_tfArray.reshape([1, resized_tfArray.shape[0],resized_tfArray.shape[1],resized_tfArray.shape[2]]);
    // console.log(reshaped_rs_tfArray.shape);
    
    let Yr = model.predict(reshaped_rs_tfArray); //tf array
    Yr = tf.squeeze(Yr); // (1,x,x,1) => (x,x)
    // let Yr_array = Yr.arraySync();
    // console.log(Yr);
    // console.log(Yr.shape);
    return Yr.greater(thresh_arr);
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
    let dst_d=resize_image_by_ratio(mat,resize_ratio);
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
function resize_image_(src,width,height){
    let dst = new cv.Mat();
    let dsize = new cv.Size(Math.round(width),Math.round(height));
    // You can try more different parameters
    cv.resize(src, dst, dsize, 0, 0, cv.INTER_AREA);
    return dst;
}
// 画像のリサイズ関数
function resize_image_by_ratio(src,ratio){
    let dst = new cv.Mat();
    // get original image size
    cv.resize(src, dst, new cv.Size(0,0),ratio,ratio, cv.INTER_NEAREST );
    return dst;
};


// RGB範囲内か否かで二値化する処理

function binarize_with_color_info(src,low_rgba,high_rgba){
    let dst = new cv.Mat();
    let low = new cv.Mat(src.rows, src.cols, src.type(),low_rgba);
    let high = new cv.Mat(src.rows, src.cols, src.type(), high_rgba);
    cv.inRange(src, low, high, dst);
        return dst;
};

function mat_info_(src){
    let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
    if (src.isContinuous()) {
        for (let row = 0; row < src.rows; ++row) {
            for (let col = 0; col < src.cols; ++col) {
                let R = src.data[row * src.cols * src.channels() + col * src.channels()];
                // let G = src.data[row * src.cols * src.channels() + col * src.channels() + 1];
                // let B = src.data[row * src.cols * src.channels() + col * src.channels() + 2];
                // let A = src.data[row * src.cols * src.channels() + col * src.channels() + 3];
                console.log(R);
            }
        }
    }
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

function erode_image(src,kernel){
    let M = cv.Mat.ones(kernel,kernel, cv.CV_8U);
    let dst = new cv.Mat();
    let anchor = new cv.Point(-1, -1);
    cv.erode(src, dst, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    return dst;
};


function dilate_image(src,kernel){
    let M = cv.Mat.ones(kernel,kernel, cv.CV_8U);
    let dst = new cv.Mat();
    let anchor = new cv.Point(-1, -1);
    cv.dilate(src, dst, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    return dst;
};


function processing(src){

    //  resize (unify size)
    let resize_ratio=DEFINED_HEIGHT/src.rows;
    src=resize_image_by_ratio(src,resize_ratio);

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