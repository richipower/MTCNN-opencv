//
//  main.cpp
//  MTCNN
//
//  Created by Ricardo Rodriguez on 5/11/18.
//  Copyright © 2018 Ricardo Rodriguez. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;


struct FaceBox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float regression[4];
    float point_coords[10];
    float score;

    cv::Rect getBox() { return Rect(x1,y1,x2-x1,y2-y1); }

};



std::vector<FaceBox> composeFaces(cv::Mat _inputScoresBlob, cv::Mat _inputRegressionsBlob, float _currentScale, float _threshold)
{
    int stride = 2;
    int windowSide = 12;
    Mat scores_data = _inputScoresBlob.reshape(1,2);
    Mat regressions_data = _inputRegressionsBlob.reshape(1,4);

    std::vector<FaceBox> faces_1;

    int width = _inputScoresBlob.size[3];

    for (int idx = 0; idx < scores_data.cols; idx++)
    {
        float score = scores_data.at<float>(1, idx); // - row, column
        if (score < _threshold)
            continue;

        FaceBox face;
        face.score = score;
        float y = std::floor(idx/width);
        float x = std::floor(idx - (y*width));
        face.x1 = std::floor((stride * x + 1)/_currentScale);
        face.y1 = std::floor((stride * y + 1)/_currentScale);
        face.x2 = std::floor((stride * x + windowSide)/_currentScale);
        face.y2 = std::floor((stride * y + windowSide)/_currentScale);
        face.regression[0] = regressions_data.at<float>(0, idx);
        face.regression[1] = regressions_data.at<float>(1, idx);
        face.regression[2] = regressions_data.at<float>(2, idx);
        face.regression[3] = regressions_data.at<float>(3, idx);

        faces_1.push_back(face);
    }


    return faces_1;
}


void applyRegression(std::vector<FaceBox> &_faces, bool addOne)
{
    for(auto &face : _faces)
    {
        float box_width = face.x2 - face.x1 + addOne;
        float box_height = face.y2 - face.y1 + addOne;
        face.x1 = face.x1 + face.regression[1] * box_width;
        face.y1 = face.y1 + face.regression[0] * box_height;
        face.x2 = face.x2 + face.regression[3] * box_width;
        face.y2 = face.y2 + face.regression[2] * box_height;
    }
}


void buildBoxes(std::vector<FaceBox> &_faces)
{
    for(auto &face : _faces)
    {
        float box_width = face.x2 - face.x1;
        float box_height = face.y2 - face.y1;
        float side = std::max(box_width, box_height);
        face.x1 = (int)(face.x1 + (box_width - side) * 0.5f);
        face.y1 = (int)(face.y1 + (box_height - side) * 0.5f);
        face.x2 = (int)(face.x1 + side);
        face.y2 = (int)(face.y1 + side);
    }
}



void fixBoxBoundaries(cv::Mat _img, FaceBox &_face)
{
    if(_face.x1 < 0)
        _face.x1 = 0;

    if(_face.x2 >= _img.cols)
        _face.x2 = _img.cols - 1;

    if(_face.y1 < 0)
        _face.y1 = 0;

    if(_face.y2 >= _img.rows)
        _face.y2 = _img.rows - 1;
}

void transformBoxes(std::vector<FaceBox> &_faces)
{
    for(auto &face : _faces)
    {
        std::swap(face.x1, face.y1);
        std::swap(face.x2, face.y2);

//        for (int p = 0; p < 5; p++)
//            std::swap(face.point_coords[2 * p], face.point_coords[2 * p + 1]);

    }
}


std::vector<Mat> generate_DataList(cv::Mat _img, std::vector<FaceBox> _faces, int _netSize)
{
    std::vector<Mat> dataList;

    for(auto face : _faces)
    {
        fixBoxBoundaries(_img, face);
        Mat tempMat = _img(face.getBox()).clone();
        resize(tempMat, tempMat, Size(_netSize, _netSize));
        dataList.push_back(tempMat);
    }

    return dataList;
}


std::vector<FaceBox> MTCNN_1(cv::Mat _img, cv::dnn::Net _net, float _minFaceSize, float _scaleFactor, float _threshold)
{
    std::vector<FaceBox> faces_1;
    int netSize = 12;

    float maxFaceSize = std::min(_img.rows, _img.cols);
    float faceSize = _minFaceSize;

    while (faceSize <= maxFaceSize)
    {
        float currentScale = (float)netSize / faceSize;

        int imgHeight = std::ceil(_img.rows * currentScale);
        int imgWidth = std::ceil(_img.cols * currentScale);
        cv::Mat inputMat;
        cv::resize(_img, inputMat, cv::Size(imgWidth, imgHeight), 0, 0, cv::INTER_AREA);

        Mat inputBlob = blobFromImage(inputMat, 1.0, Size(), Scalar(), false, false);

        _net.setInput(inputBlob);

        std::vector<String> outputBlobNames = { "prob1", "conv4-2" };
        std::vector<cv::Mat> outputBlobs;

        _net.forward(outputBlobs, outputBlobNames);
        cv::Mat scoresBlob = outputBlobs[0];
        cv::Mat regressionsBlob = outputBlobs[1];

        std::vector<FaceBox> faces = composeFaces(scoresBlob, regressionsBlob, currentScale, _threshold);

        if (!faces.empty())
            faces_1.insert(faces_1.end(), faces.begin(), faces.end());

        faceSize = faceSize / _scaleFactor;
    }

    applyRegression(faces_1, false);
    buildBoxes(faces_1);

    return faces_1;
}



std::vector<FaceBox> MTCNN_2(cv::Mat _img, cv::dnn::Net _net, std::vector<FaceBox> _faces, float _threshold)
{
    int netSize = 24;
    std::vector<FaceBox> faces_2;

    vector<Mat> dataList = generate_DataList(_img, _faces, netSize);

    Mat inputBlob = blobFromImages(dataList, 1.0, Size(), Scalar(), false, false);
    int num_imgs = inputBlob.size[0];

    _net.setInput(inputBlob);

    vector<String> blobNames;
    blobNames.push_back("conv5-2");
    blobNames.push_back("prob1");

    vector<Mat> R_net_outputs; // - [0]-regressions, [1]-scores
    _net.forward(R_net_outputs, blobNames);
    Mat regressionsBlob = R_net_outputs[0];
    Mat scoresBlob = R_net_outputs[1];

    Mat score_data = scoresBlob.reshape(1,2);
    Mat regression_data = regressionsBlob.reshape(1,4);

    for(int imgNr=0; imgNr<num_imgs; imgNr++)
    {
        float score = score_data.at<float>(1, imgNr);
        if(score < _threshold)
            continue;

        FaceBox face = _faces[imgNr];
        face.regression[0] = regression_data.at<float>(0, imgNr);
        face.regression[1] = regression_data.at<float>(1, imgNr);
        face.regression[2] = regression_data.at<float>(2, imgNr);
        face.regression[3] = regression_data.at<float>(3, imgNr);
        face.score = score;
        faces_2.push_back(face);
    }

    applyRegression(faces_2, true);
    buildBoxes(faces_2);

    return faces_2;
}


std::vector<FaceBox> MTCNN_3(cv::Mat _img, cv::dnn::Net _net, std::vector<FaceBox> _faces, float _threshold)
{
    int netSize = 48;
    std::vector<FaceBox> faces_3;

    vector<Mat> dataList = generate_DataList(_img, _faces, netSize);
    Mat inputBlob = blobFromImages(dataList, 1.0, Size(), Scalar(), false, false);
    int num_imgs = inputBlob.size[0];

    _net.setInput(inputBlob);

    vector<String> blobNames;
    blobNames.push_back("conv6-2");        // - regression
    blobNames.push_back("prob1");          // - scores
    blobNames.push_back("conv6-3");        // - points

    vector<Mat> O_net_outputs; // - [0]-regressions, [1]-scores, [2]-points
    _net.forward(O_net_outputs, blobNames);

    Mat regressionsBlob = O_net_outputs[0];
    Mat scoresBlob = O_net_outputs[1];
    Mat pointsBlob = O_net_outputs[2];

    Mat regression_data = regressionsBlob.reshape(1,4);
    Mat score_data = scoresBlob.reshape(1,2);
    Mat points_data = pointsBlob.reshape(1,10);


    for(int imgNr=0; imgNr<num_imgs; imgNr++)
    {
        float score = score_data.at<float>(1, imgNr);
        if (score < _threshold)
            continue;

        FaceBox face = _faces[imgNr];
        face.regression[0] = regression_data.at<float>(0, imgNr);
        face.regression[1] = regression_data.at<float>(1, imgNr);
        face.regression[2] = regression_data.at<float>(2, imgNr);
        face.regression[3] = regression_data.at<float>(3, imgNr);
        face.score = score;

//        for (int p = 0; p < 10; p++)
//        {
//            face.point_coords[2*p] = face.x1 + points_data.at<float>(imgNr, p + 5) * (face.x2 - face.x1 + 1) - 1;
//            face.point_coords[2*p+1] = face.y1 + points_data.at<float>(imgNr, p) * (face.y2 - face.y1 + 1) - 1;
//        }

        faces_3.push_back(face);
    }

    applyRegression(faces_3, true);
    buildBoxes(faces_3);

    return faces_3;
}





int main(int argc, char **argv)
{
    Net P_Net = readNet("../../mtcnn_models/det1.prototxt", "../../mtcnn_models/det1.caffemodel");
    Net R_Net = readNet("../../mtcnn_models/det2.prototxt", "../../mtcnn_models/det2.caffemodel");
    Net O_Net = readNet("../../mtcnn_models/det3.prototxt", "../../mtcnn_models/det3.caffemodel");

    Mat img = imread("../../imgs/jolie.jpg");
    Mat input_img = img.clone();
    cv::cvtColor(input_img, input_img, COLOR_BGR2RGB);
    input_img.convertTo(input_img, CV_32FC3);
    input_img = input_img - 127.5f;
    input_img = input_img * (1.f / 128.f);
    input_img = input_img.t();

    float pThreshold = 0.5f;
    float rThreshold = 0.5f;
    float oThreshold = 0.3f;

    cv::TickMeter tm;
    tm.reset();
    tm.start();

    std::vector<FaceBox> faces = MTCNN_1(input_img, P_Net, 60, 0.79f, pThreshold);
    faces = MTCNN_2(input_img, R_Net, faces, rThreshold);
    faces = MTCNN_3(input_img, O_Net, faces, oThreshold);

    tm.stop();
    std::cout << "Time: "<< tm.getTimeSec() << " sec at Input Size: " << img.size() << endl;

    transformBoxes(faces);

    for(auto box : faces)
        rectangle(img, box.getBox(), Scalar(0,255,0), 3);

    imshow("Output", img);
    waitKey();



}
