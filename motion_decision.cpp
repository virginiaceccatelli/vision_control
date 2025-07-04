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

# ----------------- Binary Ground Mask Calculation -----------------

class GroundSegmenter {
    public:
        GroundSegmenter(const string& model_path) {
            module = torch::jit::load(model_path);
            module.eval();
        }
    
        pair<Mat, Mat> predict_mask(const Mat& img_bgr) {
            Mat resized, rgb;
            resize(img_bgr, resized, Size(320, 320));
            cvtColor(resized, rgb, COLOR_BGR2RGB);
        
            torch::Tensor tensor = torch::from_blob(rgb.data, {1, 320, 320, 3}, torch::kByte);
            tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;
        
            torch::NoGradGuard no_grad;
            torch::Tensor output = module.forward({tensor}).toTensor();
            output = output.detach().cpu();
            if (output.dim() == 4) output = output.squeeze(0);
            if (output.dim() == 3) output = output.squeeze(0);
        
            Mat prob_map(320, 320, CV_32F);
            for (int i = 0; i < 320; ++i) {
                for (int j = 0; j < 320; ++j) {
                    prob_map.at<float>(i, j) = torch::sigmoid(output[i][j]).item<float>();
                }
            }
        
            Mat mask(320, 320, CV_8UC1);
            for (int i = 0; i < 320; ++i) {
                for (int j = 0; j < 320; ++j) {
                    mask.at<uchar>(i, j) = prob_map.at<float>(i, j) > 0.5f ? 255 : 0;
                }
            }
        
            resize(mask, mask, img_bgr.size());
            resize(prob_map, prob_map, img_bgr.size(), 0, 0, INTER_NEAREST);
        
            return {mask, prob_map};
        }        
    
    private:
        torch::jit::script::Module module;
    };
    
# ----------------- Motion Decision -----------------

    class MotionDeciderWithLasers {
    public:
        MotionDeciderWithLasers(int width, int height, int num_beams = 10) 
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

# ----------------- Main Class -----------------
    
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

	int fps = 20;
	VideoWriter out("output_video.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(w, h));
	VideoWriter heatmap_out("heatmap_output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(w, h));

    if (!out.isOpened()) {
	    cerr << "Failed to open video writer" << endl;
	    return -1;
	}
    
        MotionDeciderWithLasers decider(w, h);
        auto last_time = chrono::steady_clock::now();
    
        Mat mask; // no prefilled image
        Mat prob_map; 
	map<string, int> laser_scores;
        string decision = "waiting...";
    
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
    
            auto now = chrono::steady_clock::now();
            float elapsed = chrono::duration_cast<chrono::seconds>(now - last_time).count();
    
            if (elapsed >= 3) {
                tie(mask, prob_map) = segmenter.predict_mask(frame);
                laser_scores = decider.laser_scan(mask);
                decision = decider.decide(laser_scores);
                last_time = now;
            }
    
            if (decision != "waiting...") {
                // Visual Overlay Simple Ground Mask 
                Mat color_mask, blended, overlay;
                applyColorMap(mask, color_mask, COLORMAP_JET);
                addWeighted(frame, 0.7, color_mask, 0.3, 0, blended);
                blended.copyTo(overlay);
    
                int step = decider.width / decider.num_beams;
                int max_score = 1;
                for (const auto& [_, val] : laser_scores)
                    max_score = max(max_score, val);
    
                map<string, int> region_centers;
    
                for (int i = 0; i < decider.num_beams; ++i) {
                    string region = decider.regions[i];
                    int x_start = i * step;
                    int x_end = x_start + step;
                    int val = laser_scores[region];
                    float norm_score = static_cast<float>(val) / max_score;
                    int intensity = static_cast<int>(255 * (1.0f - norm_score));
    
                    rectangle(overlay, Point(x_start, h / 2), Point(x_end, h),
                              Scalar(intensity, intensity, intensity), FILLED);
    
                    region_centers[region] = (x_start + x_end) / 2;
                }
    
                addWeighted(overlay, 0.2, blended, 0.8, 0, blended);
    
                for (int i = 0; i < decider.num_beams; ++i) {
                    int x_start = i * step;
                    int x_end = x_start + step;
                    rectangle(blended, Point(x_start, h / 2), Point(x_end, h), Scalar(100, 100, 100), 1);
    
                    string region = decider.regions[i];
                    int val = laser_scores[region];
                    Point text_pos(x_start + 5, h - 10);
                    putText(blended, to_string(val), text_pos + Point(1, 1),
                            FONT_HERSHEY_PLAIN, 1.3, Scalar(0, 0, 0), 2); // shadow
                    putText(blended, to_string(val), text_pos,
                            FONT_HERSHEY_PLAIN, 1.3, Scalar(255, 255, 255), 1); // text
                }
    
                if (decision != "stop" && region_centers.count(decision)) {
                    int cx = region_centers[decision];
                    Point start(w / 2, h);
                    Point end(cx, h / 2);
                    line(blended, start, end, Scalar(255, 255, 255), 2);
                }
    
                Point text_pos(10, 30);
                putText(blended, "Decision: " + decision + " degrees", text_pos, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

                // Visual Overlay Heatmask
                Mat prob_map_uint8, heatmap, blended_heatmap;
                prob_map.convertTo(prob_map_uint8, CV_8UC1, 255.0);
                applyColorMap(prob_map_uint8, heatmap, COLORMAP_JET);
                addWeighted(frame, 0.6, heatmap, 0.4, 0, blended_heatmap);

                putText(blended_heatmap, "Decision: " + decision + " degrees", Point(10, 30),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
                imshow("Probability Heatmap", blended_heatmap);
                heatmap_out.write(blended_heatmap);

                imshow("Motion Decision Overlay", blended);
		        out.write(blended);
            } else {
                imshow("Motion Decision Overlay", frame); // show raw frame until prediction
            }
    
            if (waitKey(1) == 27) break; // ESC to quit
        }
    
        cap.release();
        out.release();
        heatmap_out.release();
        destroyAllWindows();
        return 0;
    }    
