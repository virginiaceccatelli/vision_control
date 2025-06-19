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

    void process_image(const std::string& image_path, GroundSegmenter& segmenter, MotionDeciderWithLasers& decider) {
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            throw std::runtime_error("Could not load image.");
        }
    
        cv::Mat mask = segmenter.predict_mask(frame);
        auto laser_scores = decider.laser_scan(mask);
        std::string decision = decider.decide(laser_scores);
    
        // Visualization
        cv::Mat color_mask;
        applyColorMap(mask, color_mask, cv::COLORMAP_JET);
        cv::Mat blended;
        addWeighted(frame, 0.7, color_mask, 0.3, 0, blended);
        cv::Mat blended_copy = blended.clone();
    
        int step = decider.width / decider.num_beams;
        int max_score = 1;
        for (const auto& kv : laser_scores) max_score = std::max(max_score, kv.second);
        int h = decider.height;
        std::map<std::string, int> region_centers;
    
        for (int i = 0; i < decider.num_beams; ++i) {
            std::string region = decider.regions[i];
            int x_start = i * step;
            int x_end = x_start + step;
            float norm_score = laser_scores[region] / static_cast<float>(max_score);
            int intensity = static_cast<int>(255 * (1 - norm_score));
    
            cv::Mat overlay = blended_copy.clone();
            cv::rectangle(overlay, Point(x_start, h/2), Point(x_end, h), Scalar(intensity, intensity, intensity), -1);
            addWeighted(overlay, 0.3, blended_copy, 0.7, 0, blended_copy);
            cv::rectangle(blended_copy, Point(x_start, h/2), Point(x_end, h), Scalar(100, 100, 100), 1);
            cv::putText(blended_copy, std::to_string(laser_scores[region]), Point(x_start + 5, h - 10),
                        cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(230, 230, 230), 1);
            region_centers[region] = (x_start + x_end) / 2;
        }
    
        if (decision != "stop" && region_centers.count(decision)) {
            int center_x = region_centers[decision];
            cv::line(blended_copy, Point(decider.width / 2, h), Point(center_x, h / 2), Scalar(255, 255, 255), 2);
        }
    
        cv::imshow("Motion Decision", blended_copy);
        cv::waitKey(0);
    
        std::cout << "Laser scores:\n";
        for (const auto& [k, v] : laser_scores)
            std::cout << "  " << k << ": " << v << '\n';
        std::cout << "Final decision: " << decision << "°" << std::endl;
    }    
    
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
    	
        cv::VideoWriter writer("output_visualization.avi", cv::VideoWriter::fourcc('M','J','P','G'), 15, Size(w, h));
        Mat overlay_frame;  

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            auto now = chrono::steady_clock::now();
            float elapsed = chrono::duration_cast<chrono::seconds>(now - last_time).count();

            if (elapsed >= 3) {
                Mat mask = segmenter.predict_mask(frame);
                auto scores = decider.laser_scan(mask);
                string decision = decider.decide(scores);
                last_time = now;

                // === Visualization ===
                Mat color_mask;
                applyColorMap(mask, color_mask, cv::COLORMAP_JET);
                Mat blended;
                addWeighted(frame, 0.7, color_mask, 0.3, 0, blended);
                overlay_frame = blended.clone();  

                int step = decider.width / decider.num_beams;
                int max_score = 1;
                for (const auto& kv : scores) max_score = std::max(max_score, kv.second);
                int h = decider.height;
                std::map<std::string, int> region_centers;

                for (int i = 0; i < decider.num_beams; ++i) {
                    std::string region = decider.regions[i];
                    int x_start = i * step;
                    int x_end = x_start + step;
                    float norm_score = scores[region] / static_cast<float>(max_score);
                    int intensity = static_cast<int>(255 * (1 - norm_score));

                    Mat overlay = overlay_frame.clone();
                    cv::rectangle(overlay, Point(x_start, h/2), Point(x_end, h), Scalar(intensity, intensity, intensity), -1);
                    addWeighted(overlay, 0.3, overlay_frame, 0.7, 0, overlay_frame);
                    cv::rectangle(overlay_frame, Point(x_start, h/2), Point(x_end, h), Scalar(100, 100, 100), 1);
                    cv::putText(overlay_frame, std::to_string(scores[region]), Point(x_start + 5, h - 10),
                                cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(230, 230, 230), 1);
                    region_centers[region] = (x_start + x_end) / 2;
                }

                if (decision != "stop" && region_centers.count(decision)) {
                    int center_x = region_centers[decision];
                    cv::line(overlay_frame, Point(decider.width / 2, h), Point(center_x, h / 2), Scalar(255, 255, 255), 2);
                }

                std::cout << "Decision: " << decision << std::endl;
            }

            if (!overlay_frame.empty()) {
                imshow("Live", overlay_frame);
                writer.write(overlay_frame);
            } else {
                imshow("Live", frame);
                writer.write(frame);
            }

            if (waitKey(1) == 27) break; // ESC to quit
        }

            
        cap.release();
            destroyAllWindows();
            return 0;
    }
