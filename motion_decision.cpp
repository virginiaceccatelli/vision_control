#include <torch/torch.h>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <map>
#include <string>

using namespace std;
using namespace cv;

class GroundSegmenter {
    public:
        GroundSegmenter(const string& model_path) {
            module = torch::jit::load(model_path);
            module.eval();
        }
    
        Mat predict_mask(const Mat& img_bgr) {
            Mat resized, rgb;
            resize(img_bgr, resized, Size(320, 320));
            cvtColor(resized, rgb, COLOR_BGR2RGB);
    
            torch::Tensor tensor = torch::from_blob(rgb.data, {1, 320, 320, 3}, torch::kByte);
            tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;
    
            torch::NoGradGuard no_grad;
            torch::Tensor output = module.forward({tensor}).toTensor();
            output = output.detach().cpu();
            if (output.dim() == 4) output = output.squeeze(0);  // remove batch dim
            if (output.dim() == 3) output = output.squeeze(0);  // remove channel dim
            std::cout << "Output shape: " << output.sizes() << std::endl;
    
            Mat mask(320, 320, CV_8UC1);
            for (int i = 0; i < 320; ++i) {
                for (int j = 0; j < 320; ++j) {
                    mask.at<uchar>(i, j) = output[i][j].item<float>() > 0.5 ? 255 : 0;
                }
            }
    
            Mat upscaled;
            resize(mask, upscaled, img_bgr.size());
            return upscaled;
        }
    
    private:
        torch::jit::script::Module module;
    };
    
    class MotionDeciderWithLasers {
    public:
        MotionDeciderWithLasers(int width, int height, int num_beams = 7)
            : width(width), height(height), num_beams(num_beams) {
            int angle_step = 180 / (num_beams - 1);
            for (int i = 0; i < num_beams; ++i)
                regions.push_back(to_string(-90 + i * angle_step));
        }
    
        map<string, int> laser_scan(const Mat& mask) {
            map<string, int> laser_scores;
            int step = width / num_beams;
            for (int i = 0; i < num_beams; ++i) {
                int x_start = i * step;
                int x_end = x_start + step;
                Rect region_rect(x_start, height / 2, step, height / 2);
                Mat region = mask(region_rect);
                laser_scores[regions[i]] = countNonZero(region);
            }
            return laser_scores;
        }
    
        string decide(const map<string, int>& scores, float threshold_ratio = 0.05) {
            int total = 0;
            for (const auto& [region, val] : scores)
                total += val;
            if (total == 0) return "stop";
    
            auto best = max_element(scores.begin(), scores.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            if (best->second < total * threshold_ratio)
                return "stop";
            return best->first;
        }
    
        int width, height;
        int num_beams;
        vector<string> regions;
    };
    
    int main() {
        string model_path = "/Users/virginiaceccatelli/Documents/vision_control/unet_ground_plane.pt";
        GroundSegmenter segmenter(model_path);
    
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Camera failed to open" << endl;
            return -1;
        }
    
        Mat frame;
        cap >> frame;
        int w = frame.cols;
        int h = frame.rows;
    
        MotionDeciderWithLasers decider(w, h);
        auto last_time = chrono::steady_clock::now();
    
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
    
            auto now = chrono::steady_clock::now();
            float elapsed = chrono::duration_cast<chrono::seconds>(now - last_time).count();
            if (elapsed >= 3) {
                Mat mask = segmenter.predict_mask(frame);
                auto scores = decider.laser_scan(mask);
                string decision = decider.decide(scores);
                cout << "Decision: " << decision << endl;
                last_time = now;
            }
    
            imshow("Live", frame);
            if (waitKey(1) == 27) break; // ESC to quit
        }
        cap.release();
        destroyAllWindows();
        return 0;
    }
