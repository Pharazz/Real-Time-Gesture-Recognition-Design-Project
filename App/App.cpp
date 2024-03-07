#include "windows.h"
#include "winerror.h"
#include "comdef.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <chrono>
#include <thread>
#include <stringapiset.h>
#include <fileapi.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "OpenCVHelper.h"
cv::ColorConversionCodes rgbColourMode;
int numChannels;
float output_thresholds[] = { 9, 5, 5, 9, 5, 9, 9, 9, 6, 9};
// input UINT8 or float32, returns float32
cv::Mat blurDep(cv::Mat src) {
 if (src.type() != CV_8U) {
 src.convertTo(src, CV_8U, 255, 0);
 }
 cv::Mat ret;
 cv::medianBlur(src, ret, 5);
 cv::adaptiveThreshold(ret, ret, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
 ret.convertTo(ret, CV_32F, 1.0 / 255, 0);
 return ret;
}
// input std::vector UINT8 or flaot32 RGB format, returns float32
std::vector<cv::Mat> backgroundSubRGB(std::vector<cv::Mat> srcVec) {
 // if something is not working as expected, try converting from bgr 2 rgb here
 std::vector<cv::Mat> retVec;
 cv::Mat fgMask;
 Ptr<BackgroundSubtractorKNN> backSubRGB = cv::createBackgroundSubtractorKNN();
 for (int i = 0; i < 30; i++) {
 if (srcVec[i].type() == CV_8UC4) {
 cv::cvtColor(srcVec[i], srcVec[i], cv::COLOR_BGRA2GRAY);
 }
 backSubRGB->apply(srcVec[i], fgMask);
 if (i > 3) {
 cv::Mat temp;
 fgMask.convertTo(temp, CV_32F, 1.0 / 255, 0);
 retVec.push_back(temp);
 }
 }
 return retVec;
}
// mat to tensor, ARGB mode default. Normalizes to float
torch::Tensor matToTensor(cv::Mat const& src, int type, at::TensorOptions options)
{
 cv::Mat normalized;
 if (type == CV_8UC4) {
 cv::cvtColor(src, normalized, rgbColourMode, numChannels);
 normalized.convertTo(normalized, CV_32F, 1.0 / 255, 0);
 torch::Tensor out = torch::zeros({ normalized.rows, normalized.cols, normalized.channels() }, options);
 memcpy(out.data_ptr(), normalized.data, 4 * out.numel());
 out = out.permute({ 2, 0, 1 }); // H, W, C --> C, H, W
 return out;
 }
 else if (type == CV_16U) {
 src.convertTo(normalized, CV_32F, 1.0 / 65535, 0);
 torch::Tensor out = torch::zeros({ normalized.rows, normalized.cols, normalized.channels() }, options);
 memcpy(out.data_ptr(), normalized.data, 4 * out.numel());
 return out;
 }
 else {
 torch::Tensor out = torch::zeros({ src.rows, src.cols, src.channels() }, options);
 memcpy(out.data_ptr(), src.data, 4 * out.numel());
 return out;
 }
}
// used to mimic std::deque behaviour in std::vector
template <typename V>
void pop_front(V& v)
{
 assert(!v.empty());
 v.erase(v.begin());
}
// gets kinect sensor and slaps dat bad boy onto the opencvframehelper. shamelessly and brazenly stolen from kinect sdk
HRESULT getSensor(INuiSensor* sensor, Microsoft::KinectBridge::OpenCVFrameHelper* frameHelper) {
 // If Kinect is already initialized, return
 if (frameHelper->IsInitialized())
 {
 return S_OK;
 }
 // Get number of Kinect sensors
 int sensorCount = 0;
 HRESULT hr = NuiGetSensorCount(&sensorCount);
 if (FAILED(hr))
 {
 return hr;
 }
 // Iterate through Kinect sensors until one is successfully initialized
 for (int i = 0; i < sensorCount; ++i)
 {
 INuiSensor* sensor = NULL;
 hr = NuiCreateSensorByIndex(i, &sensor);
 if (SUCCEEDED(hr))
 {
 hr = frameHelper->Initialize(sensor);
 if (FAILED(hr))
 {
 // Uninitialize KinectHelper to show that Kinect is not ready
 frameHelper->UnInitialize();
 }
 return hr;
 }
 }
// Report failure

return E_FAIL;
}
int main() {
 // call kinect fn's
 Microsoft::KinectBridge::OpenCVFrameHelper frameHelper; // object that handles getting frames
 INuiSensor* sensor = NULL;
 HRESULT hr = frameHelper.SetNuiInitFlags(true, true, false);
 if (FAILED(hr)) {
 std::cout << "Failed to initialize Kinect sensor. Try unplugging and replugging the Kinect." << std::endl;
 _com_error err(hr);
 std::cout << "Error message: " << err.ErrorMessage() << std::endl;
 return 0;
 }
 hr = getSensor(sensor, &frameHelper);
 if (FAILED(hr)) {
 std::cout << "Failed to acquire Kinect sensor. Ensure that a Kinect sensor is connected." << std::endl;
 _com_error err(hr);
 std::cout << "Error message: " << err.ErrorMessage() << std::endl;
 return 0;
 }
#ifdef DATA_ACQ
 char selection;
 std::cout << "Enter 'C' for colour frames or 'G' for grayscale frames." << std::endl;
 std::cin >> selection;
 if (std::tolower(selection) == 'c') {
 rgbColourMode = cv::COLOR_BGRA2RGB;
 numChannels = 3;
 std::cout << "Colour mode selected." << std::endl;
 }
 else if (std::tolower(selection) == 'g') {
 rgbColourMode = cv::COLOR_BGRA2GRAY;
 numChannels = 1;
 std::cout << "Gray scale mode selected." << std::endl;
 }
 else {
 rgbColourMode = cv::COLOR_BGRA2GRAY;
 numChannels = 1;
 std::cout << "Invalid selection. Defaulted to gray scale mode." << std::endl;
 }
 std::cout << "Enter directory path to save files." << std::endl;
 std::string path;
 std::cin >> path;
 if (!CreateDirectory(path.c_str(),NULL) && !(ERROR_ALREADY_EXISTS == GetLastError())) {
 std::cout << "Failed to create directory. Try again buster." << std::endl;
 }
#else
 rgbColourMode = cv::COLOR_BGRA2RGB;
 numChannels = 3;
#endif
 int imgWidthRGB = 640;
 int imgHeightRGB = 480;
int imgWidthDepth = 320;

int imgHeightDepth = 240;
 int framesPerGesture = 30;
 frameHelper.SetColorFrameResolution(NUI_IMAGE_RESOLUTION_640x480);
 frameHelper.SetDepthFrameResolution(NUI_IMAGE_RESOLUTION_320x240);
 // create cv mat and window
 cv::Mat matRGB(imgHeightRGB, imgWidthRGB, CV_8UC4, Scalar(0, 0, 0, 0));
 cv::Mat matDepth(imgHeightDepth, imgWidthDepth, CV_16U, Scalar(0));
 const String windowRGB = "RGB Capture";
 const String windowDepth = "Depth Capture";
 HANDLE handleRGB, handleDepth;
 frameHelper.GetColorHandle(&handleRGB);
 frameHelper.GetDepthHandle(&handleDepth);
 int eventId = WaitForSingleObject(handleRGB, 100);
 if (SUCCEEDED(frameHelper.UpdateColorFrame()))
 frameHelper.GetColorImage(&matRGB);
 eventId = WaitForSingleObject(handleRGB, 100);
 if (SUCCEEDED(frameHelper.UpdateDepthFrame()))
 frameHelper.GetDepthImage(&matDepth);
 cv::imshow(windowRGB, matRGB);
 cv::moveWindow(windowRGB, 1280, 200);
 cv::imshow(windowDepth, matDepth);
 cv::moveWindow(windowDepth, 1280, 700);
 cv::pollKey();
 // start capturing data
 using frames = std::chrono::duration<int32_t, std::ratio<1, 15>>;
 auto nextFrame = std::chrono::system_clock::now() + frames{ 0 };
 auto lastFrame = nextFrame - frames{ 1 };
 auto frameBegin = std::chrono::system_clock::now();
 auto renderBegin = std::chrono::system_clock::now();
 auto downSampleBegin = std::chrono::system_clock::now();
 auto rgbOptions = torch::TensorOptions()
 .dtype(torch::kFloat32)
 .layout(torch::kStrided)
 .device(torch::kCPU)
 .requires_grad(false);
 auto depthOptions = torch::TensorOptions()
 .dtype(torch::kFloat32)
 .layout(torch::kStrided)
 .device(torch::kCPU)
 .requires_grad(false);

#ifdef DATA_ACQ
 std::vector<torch::Tensor> rgbTensorBuffer;
 std::vector<torch::Tensor> depthTensorBuffer;
#else
 // FOR USE IN TEST
 std::vector<cv::Mat> rgbMatVec;
 std::vector<cv::Mat> rgbMatVecPreProc;
 std::vector<torch::Tensor> rgbTensorBuffer;
 std::vector<torch::Tensor> depthTensorBuffer;

 torch::DeviceType device_type;
 if (torch::cuda::is_available()) {
std::cout << "CUDA available! Using GPU." << std::endl;
device_type = torch::kCUDA;
 }
 else {
 std::cout << "Using CPU." << std::endl;
 device_type = torch::kCPU;
 }
 torch::Device device(device_type);
 torch::jit::script::Module moduleDep;
 torch::jit::script::Module moduleRGB;
 try {
 if (device_type == torch::kCUDA) {
 moduleRGB = torch::jit::load("C:\\Users\\Sherwin\\source\\repos\\KinectConverter\\modelRGB_Adam_PreProc_Res10_lr0p001_batch4SCRIPTED.pt", device);
 moduleDep = torch::jit::load("C:\\Users\\Sherwin\\source\\repos\\KinectConverter\\modelDEP_scripted.pt", device);
 }
 else {
 moduleRGB = torch::jit::load("C:\\Users\\Sherwin\\source\\repos\\KinectConverter\\Traced_modelRGBproc_Res200cpu.pt", device);
 moduleDep = torch::jit::load("C:\\Users\\Sherwin\\source\\repos\\KinectConverter\\Traced_modelDEP_PreProc_Res10cpu.pt", device);
 }

 }
 catch(const c10::Error& e){
 std::cerr << "Error loading model" << std::endl;
 return -1;
 }
 moduleRGB.eval();
 moduleDep.eval();
 // END OF USE IN TEST
#endif
 char currentCharacter = '0';

 while (1) {
#ifdef DATA_ACQ
 std::cout << "Press any key when ready to draw " << currentCharacter << " or Q to exit." << std::endl;
#endif
 bool userReady = false;
 while (userReady == false) {
 eventId = WaitForSingleObject(handleRGB, 100);
 frameBegin = std::chrono::system_clock::now();
 if (SUCCEEDED(frameHelper.UpdateColorFrame()))
 frameHelper.GetColorImage(&matRGB);
 eventId = WaitForSingleObject(handleRGB, 100);
 if (SUCCEEDED(frameHelper.UpdateDepthFrame()))
 frameHelper.GetDepthImage(&matDepth);
 cv::imshow(windowRGB, matRGB);
 cv::imshow(windowDepth, matDepth);
 int keyPress = cv::pollKey();
 if (keyPress != -1) {
 if (keyPress == 'Q' || keyPress == 'q') {
 return 0;
 }else {
 userReady = true;
 }
 }
 std::this_thread::sleep_until(nextFrame);
 // calculate time to next frame
 lastFrame = nextFrame;
 nextFrame += frames{ 1 };
 }
#ifdef DATA_ACQ
for (int i = 0; i < framesPerGesture; i++) {
for (int i = 0; true ; i++) {
#endif
 //std::cout << "Beginning frame capture. Waiting for RGB handle" << std::endl;
 eventId = WaitForSingleObject(handleRGB, 100);
 frameBegin = std::chrono::system_clock::now();
 //std::cout << "Updating colour frame." << std::endl;
 if (SUCCEEDED(frameHelper.UpdateColorFrame()))
 frameHelper.GetColorImage(&matRGB);
 //std::cout << "Beginning frame capture. Waiting for depth handle" << std::endl;
 eventId = WaitForSingleObject(handleRGB, 100);
 if (SUCCEEDED(frameHelper.UpdateDepthFrame()))
 frameHelper.GetDepthImage(&matDepth);
 //std::cout << "Updating depth frame." << std::endl;
 renderBegin = std::chrono::system_clock::now();
 cv::imshow(windowRGB, matRGB);
 cv::imshow(windowDepth, matDepth);
 cv::pollKey();
 // std::cout << "Processing time: "
 // << std::chrono::duration_cast<std::chrono::milliseconds>(renderBegin - frameBegin).count()
 // << " ms\n";
 // std::cout << "Rendering time: "
 // << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - renderBegin).count()
 // << " ms\n";
 downSampleBegin = std::chrono::system_clock::now();
 cv::Mat matRGBDownsampled;
 cv::Mat matDepthDownsampled;
 cv::resize(matRGB, matRGBDownsampled, cv::Size(), 0.25, 0.25);
 cv::resize(matDepth, matDepthDownsampled, cv::Size(), 0.5, 0.5);
#ifndef DATA_ACQ
 matDepthDownsampled = blurDep(matDepthDownsampled); // preprocess depth frame one at a time
#endif
 //cv::imshow("Downsampled RGB", matRGBDownsampled);
 //cv::moveWindow("Downsampled RGB", 1000, 200);
 cv::imshow("Downsampled Depth", matDepthDownsampled);
 cv::moveWindow("Downsampled Depth", 1000, 700);
 cv::pollKey();
#ifdef DATA_ACQ
 torch::Tensor rgbTensor = matToTensor(matRGBDownsampled, CV_8UC4, rgbOptions);
 rgbTensorBuffer.push_back(rgbTensor);
 torch::Tensor depthTensor = matToTensor(matDepthDownsampled, CV_16U, depthOptions);
#else
 rgbMatVec.push_back(matRGBDownsampled);
 torch::Tensor depthTensor = matToTensor(matDepthDownsampled, CV_32F, depthOptions);
#endif

 depthTensorBuffer.push_back(depthTensor);
 std::string millisec_since_epoch = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
#ifdef DATA_ACQ
if (rgbTensorBuffer.size() == framesPerGesture) {
std::string rgbFileName = "rgb_";
 rgbFileName += currentCharacter;
 rgbFileName += "_";
 rgbFileName += millisec_since_epoch;
 rgbFileName += ".zip";
 torch::Tensor tensorList = torch::stack(rgbTensorBuffer);
 auto bytes = torch::pickle_save(rgbTensorBuffer);
 std::ofstream fout(path + "\\" + rgbFileName, std::ios::out | std::ios::binary);
 fout.write(bytes.data(), bytes.size());
 std::cout << "Saved file " + rgbFileName + " to disk." << std::endl;
 rgbTensorBuffer.clear();
 }
 if (depthTensorBuffer.size() == framesPerGesture) {
 std::string depthFileName = "dep_";
 depthFileName += currentCharacter;
 depthFileName += "_";
 depthFileName += millisec_since_epoch;
 depthFileName += ".zip";
 torch::Tensor tensorList = torch::stack(depthTensorBuffer);
 auto bytes = torch::pickle_save(depthTensorBuffer);
 std::ofstream fout(path + "\\" + depthFileName, std::ios::out | std::ios::binary);
 fout.write(bytes.data(), bytes.size());
 std::cout << "Saved file " + depthFileName + " to disk." << std::endl;
 depthTensorBuffer.clear();
 }
#else
 // FOR USE IN TESTING

 if (rgbMatVec.size() == framesPerGesture) {
 auto rgbProcBegin = std::chrono::system_clock::now();
 rgbMatVecPreProc = backgroundSubRGB(rgbMatVec);
 for (cv::Mat img : rgbMatVecPreProc) {
 torch::Tensor rgbTensor = matToTensor(img, CV_32F, rgbOptions);
 rgbTensorBuffer.push_back(rgbTensor);
 }
 cv::imshow("Downsampled RGB", rgbMatVecPreProc[25]);
 cv::moveWindow("Downsampled RGB", 1000, 200);
 //std::cout << "RGB process time: "
 // << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - rgbProcBegin).count()
 // << " ms\n";
 }

 at::Tensor outputRGB = torch::zeros({ 1,10 });
 at::Tensor outputDep = torch::zeros({ 1,10 });
 if (rgbTensorBuffer.size() == framesPerGesture - 4) {
 torch::Tensor input = torch::stack(rgbTensorBuffer).to(device_type);
 input = torch::permute(input, { 3,0,1,2 });
 input = torch::unsqueeze(input, 0);
 outputRGB = moduleRGB.forward({ input }).toTensor();
 int predictionRGB = outputRGB.argmax(1).item().toInt();
 //std::cout << outputRGB << std::endl;
 //std::cout << "Prediction RGB: " << predictionRGB << std::endl;
 rgbTensorBuffer.clear();
 pop_front <std::vector<cv::Mat>>(rgbMatVec);
 }

 if (depthTensorBuffer.size() == framesPerGesture) {
 auto depManipBegin = std::chrono::system_clock::now();
 torch::Tensor input = torch::stack(depthTensorBuffer).to(device_type);
 input = torch::permute(input, { 3,0,1,2 });
 input = torch::unsqueeze(input, 0);
 //std::cout << "Depth tensor manipulation time: "
 // << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - depManipBegin).count()
 // << " ms\n";
 auto predictBegin = std::chrono::system_clock::now();
 outputDep = moduleDep.forward({ input }).toTensor(); 
//std::cout << "Prediction time: "
 // << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - predictBegin).count()
 // << " ms\n";
 auto predictionDep = std::get<1>(outputDep.topk(1, 1)).item<int>();
 auto predictionDepSecond = std::get<1>(outputDep.topk(2, 1))[0][1].item<int>();
 auto predictionsWrong = std::get<0>(outputDep.topk(9, 1, false));
 float difference = outputDep[0][predictionDep].item<float>() - outputDep[0][predictionDepSecond].item<float>();
 auto predictionsWrongAvg = torch::mean(predictionsWrong).item<float>();
 //std::cout << "Top prediction: " << predictionDep << " Second prediction: " << predictionDepSecond << std::endl;
 //std::cout << "Difference: " << difference << std::endl;
 //std::cout << "Average of wrong predictions: " << predictionsWrongAvg << std::endl;

 //if (difference > 10) {
 //std::cout << outputDep << std::endl;
 //std::cout << "Prediction Depth: " << predictionDep << std::endl;
 //}

 pop_front <std::vector<torch::Tensor>>(depthTensorBuffer);
 }
 at::Tensor outputMean = outputRGB.add(outputDep).mul(0.5);
 auto predictionMean = std::get<1>(outputMean.topk(1, 1)).item<int>();
 auto predictionMeanSecond = std::get<1>(outputMean.topk(2, 1))[0][1].item<int>();
 float difference = outputMean[0][predictionMean].item<float>() - outputMean[0][predictionMeanSecond].item<float>();
 auto predictionsWrong = std::get<0>(outputMean.topk(9, 1, false));
 float predictionsWrongAvg = torch::mean(predictionsWrong).item<float>();
 if (output_thresholds[predictionMean] < difference) {
 std::cout << "Prediction: " << predictionMean << std::endl;
 }
 // END OF USE IN TESTING
#endif

 // std::cout << "Downsampling time: "
 // << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - downSampleBegin).count()
 // << " ms\n";
 std::this_thread::sleep_until(nextFrame);
 //std::cout << "Frame time " << i << ": "
 // << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastFrame).count()
 // << " ms\n";
 // calculate time to next frame
 lastFrame = nextFrame;
 nextFrame += frames{ 1 };
 }
#ifdef DATA_ACQ
 if (currentCharacter >= '0' && currentCharacter < '9')
 currentCharacter++;
 else
 currentCharacter = '0';
#endif
 }
}



