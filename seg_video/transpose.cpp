#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        torch::jit::script::Module module = torch::jit::load("unet_ground_plane.pt");
        module.eval();

        cv::VideoCapture cap("example_video.mp4");  // 0 = webcam, or "video.mp4"
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video source.\n";
            return -1;
        }

        cv::Mat frame;
        while (cap.read(frame)) {
            cv::resize(frame, frame, cv::Size(320, 320));
            cv::Mat rgb;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

            torch::Tensor input_tensor = torch::from_blob(
                rgb.data, {1, 320, 320, 3}, torch::kByte
            ).permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;

            auto output = module.forward({input_tensor}).toTensor();
            output = output.squeeze().detach().cpu();

            cv::Mat mask(320, 320, CV_32F, output.data_ptr());
            cv::Mat bin_mask;
            cv::threshold(mask, bin_mask, 0.5, 1.0, cv::THRESH_BINARY);
            bin_mask.convertTo(bin_mask, CV_8U, 255);

            // Overlay
            cv::Mat color_mask;
            cv::applyColorMap(bin_mask, color_mask, cv::COLORMAP_JET);
            cv::addWeighted(frame, 0.7, color_mask, 0.3, 0, frame);

            cv::imshow("Ground Plane Detection", frame);
            if (cv::waitKey(1) == 27) break;  // ESC to quit
        }

    } catch (const c10::Error& e) {
        std::cerr << "Torch error: " << e.what() << '\n';
        return -1;
    }

    return 0;
}

