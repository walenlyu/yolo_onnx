#include "yolo.h"


Net_config yolo_nets[4] = {
	{0.5, 0.5, 0.5, "yolov5s"},
	{0.5, 0.5, 0.5,  "yolov5m"},
	{0.5, 0.5, 0.5, "yolov5l"},
	{0.5, 0.5, 0.5, "yolov5x"}

};

int main()
{
	YOLO yolo_model(yolo_nets[0]);
	//VideoCapture video("rtsp:/admin:Admin12345@192.168.0.61/Streaming/Channels/1");
	//video.open("rtsp://admin:Admin12345@192.168.0.61/Streaming/Channels/1");
	VideoCapture video(0);
	int i = 0;
	while (cv::waitKey(1) < 1) {
		Mat frame;
		video >> frame;
		if (frame.empty()) {
			cout << "frame empty" << endl;
			i++;
			if (i >= 10) {
				break;
			}
			continue;
		}
		auto total_start = std::chrono::steady_clock::now();
		float  inference_fps = yolo_model.detect(frame);
		auto total_end = std::chrono::steady_clock::now();
		float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
		
		std::ostringstream stats_ss;
		stats_ss << fixed << setprecision(2);
		stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
		auto stats = stats_ss.str();

		int baseline;
		auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
		cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
		cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

		static const string kWinName = "Deep learning object detection in OpenCV";
		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, frame);
		//cv::waitKey(1);
	}
}
