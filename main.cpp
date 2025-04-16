/*
 * Yolo_mark - GUI application for marking bounded boxes of objects in images for training Yolo v3 and v2
 * This tool helps to create annotation files for training neural networks like YOLO
 */

//---------- Standard C++ Libraries ----------
#include <cstdio>        // Standard input/output functions
#include <iostream>      // Input/output stream operations
#include <vector>        // Dynamic array container
#include <numeric>       // Numeric operations on ranges
#include <chrono>        // Time-related functionality
#include <atomic>        // Atomic operations for thread safety
#include <locale>        // Localization utilities
#include <future>        // C++11: async(); feature<>; - Asynchronous operations
#include <iomanip>       // I/O manipulators for formatting
#include <fstream>       // File stream operations
#include <algorithm>     // Algorithms for ranges (sorting, searching, etc.)

//---------- OpenCV Libraries ----------
#include <opencv2/opencv.hpp>            // Main OpenCV header
#include <opencv2/core/version.hpp>      // OpenCV version information
#include <opencv2/imgproc/imgproc.hpp>   // Image processing functions
#include <opencv2/highgui/highgui.hpp>   // GUI functions and window handling
//#include <opencv2/optflow.hpp>         // Optical flow (commented out)
#include <opencv2/video/tracking.hpp>    // Object tracking functionality

//---------- Library Configuration for OpenCV ----------
#ifdef _DEBUG
#define LIB_SUFFIX "d.lib"  // Debug library suffix
#else
#define LIB_SUFFIX ".lib"   // Release library suffix
#endif // DEBUG

// OpenCV version-specific configuration
#ifndef CV_VERSION_EPOCH   // OpenCV 3.x and 4.x
#include "opencv2/videoio/videoio.hpp"  // Video I/O functionality
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world4.lib")  // Link with the unified OpenCV 4.x library
#else                      // OpenCV 2.x
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
// Link with individual OpenCV 2.x libraries
#pragma comment(lib, "opencv_core" OPENCV_VERSION LIB_SUFFIX)
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION LIB_SUFFIX)
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION LIB_SUFFIX)
#endif

// Compatibility definition for CV_FILLED constant
#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif


using namespace cv;  // Use OpenCV namespace for convenience

//---------- Data Structures ----------

// Structure to store bounding box coordinates and object class ID
struct coord_t {
    cv::Rect_<float> abs_rect;  // Rectangle with floating-point coordinates (x, y, width, height)
    int id;                     // Object class ID corresponding to the label in obj.names file
};

//---------- Optical Flow Tracking Class ----------

/**
 * Tracker_optflow - Class for tracking bounding boxes between frames using optical flow
 * Uses Lucas-Kanade optical flow algorithm to track object positions across frames
 */
class Tracker_optflow {
public:
    const int flow_error;  // Maximum allowed error for optical flow tracking

    /**
     * Constructor - initializes the optical flow tracker
     * @param win_size Size of the search window at each pyramid level
     * @param max_level 0-based maximal pyramid level number
     * @param iterations Unused parameter (legacy)
     * @param _flow_error Maximum allowed error for optical flow tracking
     */
    Tracker_optflow(int win_size = 15, int max_level = 3, int iterations = 8000, int _flow_error = -1) :
        flow_error((_flow_error > 0) ? _flow_error : (win_size * 4))
    {
        // Create and configure the sparse optical flow algorithm
        sync_PyrLKOpticalFlow = cv::SparsePyrLKOpticalFlow::create();
        sync_PyrLKOpticalFlow->setWinSize(cv::Size(win_size, win_size));    // Search window size
        sync_PyrLKOpticalFlow->setMaxLevel(max_level);                      // Maximum pyramid level
    }

    // Image matrices and tracking data
    cv::Mat dst_grey;                                   // Destination image in grayscale
    cv::Mat prev_pts_flow, cur_pts_flow;               // Previous and current tracking points
    cv::Mat status, err;                               // Status and error matrices for tracking
    cv::Mat src_grey;                                  // Source image in grayscale
    cv::Ptr<cv::SparsePyrLKOpticalFlow> sync_PyrLKOpticalFlow;  // Optical flow algorithm

    // Bounding box tracking data
    std::vector<coord_t> cur_bbox_vec;                 // Current bounding boxes
    std::vector<bool> good_bbox_vec_flags;             // Flags indicating valid tracking

    /**
     * Updates the current bounding boxes and prepares tracking points
     * @param _cur_bbox_vec Vector of bounding boxes to track
     */
    void update_cur_bbox_vec(std::vector<coord_t> _cur_bbox_vec)
    {
        // Store the new bounding boxes and initialize tracking flags
        cur_bbox_vec = _cur_bbox_vec;
        good_bbox_vec_flags = std::vector<bool>(cur_bbox_vec.size(), true);
        cv::Mat prev_pts, cur_pts_flow;

        // Extract center points of each bounding box for tracking
        for (auto &i : cur_bbox_vec) {
            // Calculate center coordinates of the bounding box
            float x_center = (i.abs_rect.x + i.abs_rect.width / 2.0F);
            float y_center = (i.abs_rect.y + i.abs_rect.height / 2.0F);
            prev_pts.push_back(cv::Point2f(x_center, y_center));
        }

        // Prepare points for optical flow tracking
        if (prev_pts.rows == 0)
            prev_pts_flow = cv::Mat();  // No points to track
        else
            cv::transpose(prev_pts, prev_pts_flow);  // Format points for optical flow algorithm
    }


    /**
     * Updates the source image and bounding boxes for tracking
     * @param new_src_mat New source image
     * @param _cur_bbox_vec Vector of bounding boxes to track
     */
    void update_tracking_flow(cv::Mat new_src_mat, std::vector<coord_t> _cur_bbox_vec)
    {
        // Convert the source image to grayscale based on its channel count
        if (new_src_mat.channels() == 1) {
            // Already grayscale
            src_grey = new_src_mat.clone();
        }
        else if (new_src_mat.channels() == 3) {
            // BGR color image
            cv::cvtColor(new_src_mat, src_grey, cv::COLOR_BGR2GRAY, 1);
        }
        else if (new_src_mat.channels() == 4) {
            // BGRA color image with alpha channel
            cv::cvtColor(new_src_mat, src_grey, cv::COLOR_BGRA2GRAY, 1);
        }
        else {
            // Unsupported image format
            std::cerr << " Warning: new_src_mat.channels() is not: 1, 3 or 4. It is = " << new_src_mat.channels() << " \n";
            return;
        }

        // Update bounding boxes for tracking
        update_cur_bbox_vec(_cur_bbox_vec);
    }


    /**
     * Tracks objects from previous frame to current frame using optical flow
     * @param new_dst_mat New destination image to track objects in
     * @param check_error Whether to check for tracking errors
     * @return Vector of tracked bounding boxes
     */
    std::vector<coord_t> tracking_flow(cv::Mat new_dst_mat, bool check_error = true)
    {
        // Check if optical flow tracker is initialized
        if (sync_PyrLKOpticalFlow.empty()) {
            std::cout << "sync_PyrLKOpticalFlow isn't initialized \n";
            return cur_bbox_vec;
        }

        // Convert destination image to grayscale
        cv::cvtColor(new_dst_mat, dst_grey, cv::COLOR_BGR2GRAY, 1);

        // Check if source and destination images have the same dimensions
        if (src_grey.rows != dst_grey.rows || src_grey.cols != dst_grey.cols) {
            src_grey = dst_grey.clone();
            // Return current bounding boxes without tracking if dimensions don't match
            return cur_bbox_vec;
        }

        // Check if there are points to track
        if (prev_pts_flow.cols < 1) {
            return cur_bbox_vec;
        }

        // Calculate optical flow between source and destination images
        sync_PyrLKOpticalFlow->calc(src_grey, dst_grey, prev_pts_flow, cur_pts_flow, status, err);

        // Update source image for next tracking iteration
        dst_grey.copyTo(src_grey);

        // Store tracked bounding boxes
        std::vector<coord_t> result_bbox_vec;

        // Process tracking results if dimensions match
        if (err.rows == cur_bbox_vec.size() && status.rows == cur_bbox_vec.size())
        {
            for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
            {
                // Get current and previous tracking points
                cv::Point2f cur_key_pt = cur_pts_flow.at<cv::Point2f>(0, i);
                cv::Point2f prev_key_pt = prev_pts_flow.at<cv::Point2f>(0, i);

                // Calculate movement vector
                float moved_x = cur_key_pt.x - prev_key_pt.x;
                float moved_y = cur_key_pt.y - prev_key_pt.y;

                // Check if tracking is valid based on movement and error
                if (abs(moved_x) < 100 && abs(moved_y) < 100 && good_bbox_vec_flags[i])
                    if (err.at<float>(0, i) < flow_error && status.at<unsigned char>(0, i) != 0 &&
                        ((float)cur_bbox_vec[i].abs_rect.x + moved_x) > 0 && ((float)cur_bbox_vec[i].abs_rect.y + moved_y) > 0)
                    {
                        // Update bounding box position
                        cur_bbox_vec[i].abs_rect.x += moved_x;
                        cur_bbox_vec[i].abs_rect.y += moved_y;
                        result_bbox_vec.push_back(cur_bbox_vec[i]);
                    }
                    else good_bbox_vec_flags[i] = false;  // Mark tracking as failed
                else good_bbox_vec_flags[i] = false;      // Mark tracking as failed
            }
        }

        // Update previous points for next tracking iteration
        prev_pts_flow = cur_pts_flow.clone();

        return result_bbox_vec;
    }

};

//---------- Global Variables for UI Interaction ----------

// Mouse interaction state variables (thread-safe atomic variables)
std::atomic<bool> right_button_click;      // Right mouse button is being pressed
std::atomic<int> move_rect_id;             // ID of the rectangle being moved
std::atomic<bool> move_rect;               // Rectangle is being moved
std::atomic<bool> clear_marks;             // Clear all marks from current image
std::atomic<bool> copy_previous_marks(false);       // Copy marks from previous image
std::atomic<bool> tracker_copy_previous_marks(false); // Track objects from previous image

// UI state variables
std::atomic<bool> show_help;               // Show help text
std::atomic<bool> exit_flag(false);        // Exit the application

// Marking appearance settings
std::atomic<int> mark_line_width(2);       // Width of bounding box lines (default: 2 pixels)
const int MAX_MARK_LINE_WIDTH = 3;         // Maximum allowed line width
std::atomic<bool> show_mark_class(true);   // Show class ID and name on bounding boxes
std::atomic<bool> delete_selected(false);  // Delete the selected bounding box

// Selection coordinates and state
std::atomic<int> x_start, y_start;         // Starting coordinates of selection
std::atomic<int> x_end, y_end;             // Ending coordinates of selection
std::atomic<int> x_size, y_size;           // Size of selection rectangle
std::atomic<bool> draw_select;             // Currently drawing a selection
std::atomic<bool> selected;                // Selection has been completed
std::atomic<bool> undo;                    // Undo last action

// Navigation variables
std::atomic<int> add_id_img;               // Image navigation direction
Rect prev_img_rect(0, 0, 50, 100);         // Rectangle for previous image button
Rect next_img_rect(1280 - 50, 0, 50, 100); // Rectangle for next image button


/**
 * Mouse event callback function for handling user interactions
 * @param event Type of mouse event (click, move, etc.)
 * @param x X-coordinate of mouse position
 * @param y Y-coordinate of mouse position
 * @param flags Additional flags for the event
 * @param user_data User-provided data (unused)
 */
void callback_mouse_click(int event, int x, int y, int flags, void* user_data)
{
    // Handle double left-click
    if (event == cv::EVENT_LBUTTONDBLCLK)
    {
        std::cout << "cv::EVENT_LBUTTONDBLCLK \n";
    }
    // Handle left mouse button press
    else if (event == cv::EVENT_LBUTTONDOWN)
    {
        // Start drawing selection rectangle
        draw_select = true;
        selected = false;
        x_start = x;
        y_start = y;

        // Check if clicking on navigation buttons
        if (prev_img_rect.contains(Point2i(x, y)))
            add_id_img = -1;  // Previous image button
        else if (next_img_rect.contains(Point2i(x, y)))
            add_id_img = 1;   // Next image button
        else
            add_id_img = 0;   // Not on navigation buttons
    }
    // Handle left mouse button release
    else if (event == cv::EVENT_LBUTTONUP)
    {
        // Calculate selection rectangle size
        x_size = abs(x - x_start);
        y_size = abs(y - y_start);
        x_end = max(x, 0);
        y_end = max(y, 0);

        // Complete selection
        draw_select = false;
        selected = true;
    }
    // Handle right mouse button press (for moving bounding boxes)
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        right_button_click = true;
        x_start = x;
        y_start = y;
        std::cout << "cv::EVENT_RBUTTONDOWN \n";
    }
    // Handle right mouse button release
    else if (event == cv::EVENT_RBUTTONUP)
    {
        right_button_click = false;
        move_rect = true;  // Complete the move operation
    }
    // Handle right mouse button double-click
    else if (event == cv::EVENT_RBUTTONDBLCLK)
    {
        std::cout << "cv::EVENT_RBUTTONDBLCLK \n";
    }
    // Handle mouse movement
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        // Update current mouse position
        x_end = max(x, 0);
        y_end = max(y, 0);
    }
}

/**
 * Custom locale class to ensure decimal points are represented as periods ('.')
 * This ensures consistent decimal formatting regardless of system locale settings
 */
class comma : public std::numpunct<char> {
public:
	comma() : std::numpunct<char>() {}
protected:
	// Override the decimal point character to always be a period
	char do_decimal_point() const { return '.';	}
};


/**
 * Main function - entry point of the application
 * Handles command-line arguments, loads images and annotations, and runs the marking interface
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return Exit code (0 for success, non-zero for errors)
 */
int main(int argc, char *argv[])
{
	try
	{
		// Set locale to ensure consistent decimal point formatting
		std::locale loccomma(std::locale::classic(), new comma);
		std::locale::global(loccomma);

		// Default path for images
		std::string images_path = "./";

		// Parse command line arguments
		if (argc >= 2) {
			images_path = std::string(argv[1]);         // Path to images directory
		}
		else {
			// Show usage information if no arguments provided
			std::cout << "Usage: [path_to_images] [train.txt] [obj.names] \n";
			return 0;
		}

		// Set default filenames based on images path
		std::string train_filename = images_path + "train.txt";  // File to store list of training images
		std::string synset_filename = images_path + "obj.names"; // File containing object class names

		// Override default filenames if provided in command line
		if (argc >= 3) {
			train_filename = std::string(argv[2]);      // Custom train.txt path
		}

		if (argc >= 4) {
			synset_filename = std::string(argv[3]);      // Custom obj.names path
		}

        // Initialize optical flow tracker for tracking objects between frames
        Tracker_optflow tracker_optflow;
        cv::Mat optflow_img;  // Image used for optical flow tracking

		// Special mode: Capture frames from video file
		// This mode is activated when train_filename is "cap_video" or "cap_video_backward"
		// It extracts frames from a video file at regular intervals
		if (argc >= 4 && (train_filename == "cap_video" || train_filename == "cap_video_backward")) {
			// Use the synset_filename as the video file path
			const std::string videofile = synset_filename;

			// Open the video file
			cv::VideoCapture cap(videofile);
				// Get frames per second based on OpenCV version
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x and 4.x
            const int fps = cap.get(cv::CAP_PROP_FPS);
#else                        // OpenCV 2.x
            const int fps = cap.get(CV_CAP_PROP_FPS);
#endif
            // Initialize counters for frame extraction
            int frame_counter = 0, image_counter = 0;

            // Check if we're extracting frames in reverse order
            int backward = (train_filename == "cap_video_backward") ? 1 : 0;
            if (backward) image_counter = 99999999; // Start from a high number and count down
			// Set frame extraction interval (default: every 50 frames)
			float save_each_frames = 50;

			// Override interval if provided as 5th argument
			if (argc >= 5) save_each_frames = std::stoul(std::string(argv[4]));

			// Extract filename from path (handling both Windows and Unix-style paths)
			int pos_filename = 0;
			if ((1 + videofile.find_last_of("\\")) < videofile.length()) pos_filename = 1 + videofile.find_last_of("\\"); // Windows path
			if ((1 + videofile.find_last_of("/")) < videofile.length()) pos_filename = std::max(pos_filename, 1 + (int)videofile.find_last_of("/")); // Unix path

			// Get filename and filename without extension
			std::string const filename = videofile.substr(pos_filename);
			std::string const filename_without_ext = filename.substr(0, filename.find_last_of("."));

			// Process video frames
			for (cv::Mat frame; cap >> frame, cap.isOpened() && !frame.empty();) {
				cv::imshow("video cap to frames", frame);
					// Wait for key press with OpenCV version-specific handling
#ifndef CV_VERSION_EPOCH
				int pressed_key = cv::waitKeyEx(20);	// OpenCV 3.x and 4.x
#else
				int pressed_key = cv::waitKey(20);		// OpenCV 2.x
#endif
				// Exit on ESC key press (handles different key codes across OpenCV versions)
				if (pressed_key == 27 || pressed_key == 1048603) break;  // ESC - exit (OpenCV 2.x / 3.x)
				// Save frame when counter reaches threshold
				if (frame_counter++ >= save_each_frames) {
					// Reset counter
					frame_counter = 0;

					// Format image counter with leading zeros
                    std::stringstream image_counter_ss;
                    image_counter_ss << std::setw(8) << std::setfill('0') << image_counter;

                    // Update counter based on direction (forward or backward)
                    if (backward) image_counter--;
                    else image_counter++;
					// Create output filename
					std::string img_name = images_path + "/" + filename_without_ext + "_" + image_counter_ss.str() + ".jpg";
					std::cout << "saved " << img_name << std::endl;

					// Save frame as image
					cv::imwrite(img_name, frame);
				}
			}
			// Exit after processing video
			exit(0);
		}

		bool show_mouse_coords = false;
		std::vector<std::string> filenames_in_folder;
		//glob(images_path, filenames_in_folder); // void glob(String pattern, std::vector<String>& result, bool recursive = false);
		cv::String images_path_cv = images_path;
		std::vector<cv::String> filenames_in_folder_cv;
		glob(images_path_cv, filenames_in_folder_cv); // void glob(String pattern, std::vector<String>& result, bool recursive = false);
		for (auto &i : filenames_in_folder_cv)
			filenames_in_folder.push_back(i);

		std::vector<std::string> jpg_filenames_path;
		std::vector<std::string> jpg_filenames;
		std::vector<std::string> jpg_filenames_without_ext;
		std::vector<std::string> image_ext;
		std::vector<std::string> txt_filenames;
		std::vector<std::string> jpg_in_train;
		std::vector<std::string> synset_txt;

        // image-paths to txt-paths
		for (auto &i : filenames_in_folder)
		{
			int pos_filename = 0;
			if ((1 + i.find_last_of("\\")) < i.length()) pos_filename = 1 + i.find_last_of("\\");
			if ((1 + i.find_last_of("/")) < i.length()) pos_filename = std::max(pos_filename, 1 + (int)i.find_last_of("/"));


			std::string const filename = i.substr(pos_filename);
			std::string const ext = i.substr(i.find_last_of(".") + 1);
			std::string const filename_without_ext = filename.substr(0, filename.find_last_of("."));

			if (ext == "jpg" || ext == "JPG" ||
				ext == "jpeg" || ext == "JPEG" ||
				ext == "bmp" || ext == "BMP" ||
				ext == "png" || ext == "PNG" ||
				ext == "ppm" || ext == "PPM")
			{
				jpg_filenames_without_ext.push_back(filename_without_ext);
				image_ext.push_back(ext);
				jpg_filenames.push_back(filename);
				jpg_filenames_path.push_back(i);
			}
			if (ext == "txt") {
				txt_filenames.push_back(filename_without_ext);
			}
		}
		std::sort(jpg_filenames.begin(), jpg_filenames.end());
		std::sort(jpg_filenames_path.begin(), jpg_filenames_path.end());
		std::sort(txt_filenames.begin(), txt_filenames.end());

		if (jpg_filenames.size() == 0) {
			std::cout << "Error: Image files not found by path: " << images_path << std::endl;
			return 0;
		}

		// check whether there are files with the same names (but different extensions)
		{
			auto sorted_names_without_ext = jpg_filenames_without_ext;
			std::sort(sorted_names_without_ext.begin(), sorted_names_without_ext.end());
			for (size_t i = 1; i < sorted_names_without_ext.size(); ++i) {
				if (sorted_names_without_ext[i - 1] == sorted_names_without_ext[i]) {
					std::cout << "Error: Can't create " << sorted_names_without_ext[i] <<
						".txt file for several images with different extensions but with the same filename: "
						<< sorted_names_without_ext[i] << std::endl;
					// print duplicate images
					for (size_t k = 0; k < jpg_filenames_without_ext.size(); ++k) {
						if (jpg_filenames_without_ext[k] == sorted_names_without_ext[i]) {
							std::cout << jpg_filenames_without_ext[k] << "." << image_ext[k] << std::endl;
						}
					}
					return 0;
				}
			}
		}

		// intersect jpg & txt
		std::vector<std::string> intersect_filenames(jpg_filenames.size());
		std::vector<std::string> difference_filenames(jpg_filenames.size());
		std::vector<std::string> intersect_ext;
		std::vector<std::string> difference_ext;

		auto dif_it_end = std::set_difference(jpg_filenames_without_ext.begin(), jpg_filenames_without_ext.end(),
			txt_filenames.begin(), txt_filenames.end(),
			difference_filenames.begin());
		difference_filenames.resize(dif_it_end - difference_filenames.begin());

		auto inter_it_end = std::set_intersection(jpg_filenames_without_ext.begin(), jpg_filenames_without_ext.end(),
			txt_filenames.begin(), txt_filenames.end(),
			intersect_filenames.begin());
		intersect_filenames.resize(inter_it_end - intersect_filenames.begin());

		// get intersect extensions for intersect_filenames
		for (auto &i : intersect_filenames) {
			size_t ext_index = find(jpg_filenames_without_ext.begin(), jpg_filenames_without_ext.end(), i) - jpg_filenames_without_ext.begin();
			intersect_ext.push_back(image_ext[ext_index]);
		}

		// get difference extensions for intersect_filenames
		for (auto &i : difference_filenames) {
			size_t ext_index = find(jpg_filenames_without_ext.begin(), jpg_filenames_without_ext.end(), i) - jpg_filenames_without_ext.begin();
			difference_ext.push_back(image_ext[ext_index]);
		}

		txt_filenames.clear();
		for (auto &i : intersect_filenames) {
			txt_filenames.push_back(i + ".txt");
		}

		int image_list_count = max(1, (int)jpg_filenames_path.size() - 1);

		// store train.txt
		std::ofstream ofs_train(train_filename, std::ios::out | std::ios::trunc);
		if (!ofs_train.is_open()) {
			throw(std::runtime_error("Can't open file: " + train_filename));
		}

		for (size_t i = 0; i < intersect_filenames.size(); ++i) {
			ofs_train << images_path << "/" << intersect_filenames[i] << "." << intersect_ext[i] << std::endl;
		}
		ofs_train.flush();
		std::cout << "File opened for output: " << train_filename << std::endl;


		// load synset.txt
		{
			std::ifstream ifs(synset_filename);
			if (!ifs.is_open()) {
				throw(std::runtime_error("Can't open file: " + synset_filename));
			}

			for (std::string line; getline(ifs, line);)
				synset_txt.push_back(line);
		}
		std::cout << "File loaded: " << synset_filename << std::endl;

		Mat preview(Size(100, 100), CV_8UC3);
		Mat full_image(Size(1280, 720), CV_8UC3);
		Mat frame(Size(full_image.cols, full_image.rows + preview.rows), CV_8UC3);

		Rect full_rect_dst(Point2i(0, preview.rows), Size(frame.cols, frame.rows - preview.rows));
		Mat full_image_roi = frame(full_rect_dst);

		size_t const preview_number = frame.cols / preview.cols;


        // labels on the current image
		std::vector<coord_t> current_coord_vec;
		Size current_img_size;


		std::string const window_name = "Marking images";
		namedWindow(window_name, WINDOW_NORMAL);
		resizeWindow(window_name, 1280, 720);
		imshow(window_name, frame);
		moveWindow(window_name, 0, 0);
		setMouseCallback(window_name, callback_mouse_click);

		bool next_by_click = false;
		bool marks_changed = false;

		int old_trackbar_value = -1, trackbar_value = 0;
		std::string const trackbar_name = "image num";
		int tb_res = createTrackbar(trackbar_name, window_name, &trackbar_value, image_list_count);

		int old_current_obj_id = -1, current_obj_id = 0;
		std::string const trackbar_name_2 = "object id";
		int const max_object_id = (synset_txt.size() > 0) ? synset_txt.size() : 20;
		int tb_res_2 = createTrackbar(trackbar_name_2, window_name, &current_obj_id, max_object_id);


		do {
			//trackbar_value = min(max(0, trackbar_value), (int)jpg_filenames_path.size() - 1);

            // selected new image
			if (old_trackbar_value != trackbar_value || exit_flag)
			{
				trackbar_value = min(max(0, trackbar_value), (int)jpg_filenames_path.size() - 1);
				setTrackbarPos(trackbar_name, window_name, trackbar_value);
				frame(Rect(0, 0, frame.cols, preview.rows)) = Scalar::all(0);
                move_rect_id = -1;

				// save current coords
				if (old_trackbar_value >= 0) // && current_coord_vec.size() > 0) // Yolo v2 can processes background-image without objects
				{
					try
					{
						std::string const jpg_filename = jpg_filenames[old_trackbar_value];
						std::string const filename_without_ext = jpg_filename.substr(0, jpg_filename.find_last_of("."));
						std::string const txt_filename = filename_without_ext + ".txt";
						std::string const txt_filename_path = images_path + "/" + txt_filename;

						std::cout << "txt_filename_path = " << txt_filename_path << std::endl;

						std::ofstream ofs(txt_filename_path, std::ios::out | std::ios::trunc);
						ofs << std::fixed;

						// store coords to [image name].txt
						for (auto &i : current_coord_vec)
						{
							float const relative_center_x = (float)(i.abs_rect.x + i.abs_rect.width / 2) / full_image_roi.cols;
							float const relative_center_y = (float)(i.abs_rect.y + i.abs_rect.height / 2) / full_image_roi.rows;
							float const relative_width = (float)i.abs_rect.width / full_image_roi.cols;
							float const relative_height = (float)i.abs_rect.height / full_image_roi.rows;

							if (relative_width <= 0) continue;
							if (relative_height <= 0) continue;
							if (relative_center_x <= 0) continue;
							if (relative_center_y <= 0) continue;

							ofs << i.id << " " <<
								relative_center_x << " " << relative_center_y << " " <<
								relative_width << " " << relative_height << std::endl;
						}

						// store [path/image name.jpg] to train.txt
						auto it = std::find(difference_filenames.begin(), difference_filenames.end(), filename_without_ext);
						if (it != difference_filenames.end())
						{
							ofs_train << images_path << "/" << jpg_filename << std::endl;
							ofs_train.flush();

							size_t new_size = std::remove(difference_filenames.begin(), difference_filenames.end(), filename_without_ext) -
								difference_filenames.begin();
							difference_filenames.resize(new_size);
						}
					}
					catch (...) { std::cout << " Exception when try to write txt-file \n"; }
				}

				// show preview images
				for (size_t i = 0; i < preview_number && (i + trackbar_value) < jpg_filenames_path.size(); ++i)
				{
					Mat img = imread(jpg_filenames_path[trackbar_value + i]);
					// check if the image has been loaded successful to prevent crash
					if (img.cols == 0)
					{
						continue;
					}
					resize(img, preview, preview.size());
					int const x_shift = i*preview.cols + prev_img_rect.width;
					Rect rect_dst(Point2i(x_shift, 0), preview.size());
					Mat dst_roi = frame(rect_dst);
					preview.copyTo(dst_roi);
					//rectangle(frame, rect_dst, Scalar(200, 150, 200), 2);
					putText(dst_roi, jpg_filenames[trackbar_value + i], Point2i(0, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar::all(255));

					if (i == 0)
					{
                        optflow_img = img;
						resize(img, full_image, full_rect_dst.size());
						full_image.copyTo(full_image_roi);
						current_img_size = img.size();

						try {
							std::string const jpg_filename = jpg_filenames[trackbar_value];
							std::string const txt_filename = jpg_filename.substr(0, jpg_filename.find_last_of(".")) + ".txt";
							//std::cout << (images_path + "/" + txt_filename) << std::endl;
							std::ifstream ifs(images_path + "/" + txt_filename);
                            if (copy_previous_marks) copy_previous_marks = false;
                            else if (tracker_copy_previous_marks) {
                                tracker_copy_previous_marks = false;
                                current_coord_vec = tracker_optflow.tracking_flow(img, false);
                            }
                            else current_coord_vec.clear();

							for (std::string line; getline(ifs, line);)
							{
								std::stringstream ss(line);
								coord_t coord;
								coord.id = -1;
								ss >> coord.id;
								if (coord.id < 0) continue;
								float relative_coord[4] = { -1, -1, -1, -1 };  // rel_center_x, rel_center_y, rel_width, rel_height
								for (size_t i = 0; i < 4; i++) if(!(ss >> relative_coord[i])) continue;
								for (size_t i = 0; i < 4; i++) if (relative_coord[i] < 0) continue;
								coord.abs_rect.x = (relative_coord[0] - relative_coord[2] / 2) * (float)full_image_roi.cols;
								coord.abs_rect.y = (relative_coord[1] - relative_coord[3] / 2) * (float)full_image_roi.rows;
								coord.abs_rect.width = relative_coord[2] * (float)full_image_roi.cols;
								coord.abs_rect.height = relative_coord[3] * (float)full_image_roi.rows;

								current_coord_vec.push_back(coord);
							}
						}
						catch (...) { std::cout << " Exception when try to read txt-file \n"; }
					}

					std::string const jpg_filename = jpg_filenames[trackbar_value + i];
					std::string const filename_without_ext = jpg_filename.substr(0, jpg_filename.find_last_of("."));
                    // green check-mark on the preview image if there is a lebel txt-file for this image
					if (!std::binary_search(difference_filenames.begin(), difference_filenames.end(), filename_without_ext))
					{
						line(dst_roi, Point2i(80, 88), Point2i(85, 93), Scalar(20, 70, 20), 5);
						line(dst_roi, Point2i(85, 93), Point2i(93, 85), Scalar(20, 70, 20), 5);

						line(dst_roi, Point2i(80, 88), Point2i(85, 93), Scalar(50, 200, 100), 2);
						line(dst_roi, Point2i(85, 93), Point2i(93, 85), Scalar(50, 200, 100), 2);
					}

				}
				std::cout << " trackbar_value = " << trackbar_value << std::endl;

				old_trackbar_value = trackbar_value;

				marks_changed = false;

				rectangle(frame, prev_img_rect, Scalar(100, 100, 100), CV_FILLED);
				rectangle(frame, next_img_rect, Scalar(100, 100, 100), CV_FILLED);
			}

			trackbar_value = min(max(0, trackbar_value), (int)jpg_filenames_path.size() - 1);

			// highlight prev img
			for (size_t i = 0; i < preview_number && (i + trackbar_value) < jpg_filenames_path.size(); ++i)
			{
				int const x_shift = i*preview.cols + prev_img_rect.width;
				Rect rect_dst(Point2i(x_shift, 0), Size(preview.cols - 2, preview.rows));
				Scalar color(100, 70, 100);
				if (i == 0) color = Scalar(250, 120, 150);
				if (y_end < preview.rows && i == (x_end - prev_img_rect.width) / preview.cols) color = Scalar(250, 200, 200);
				rectangle(frame, rect_dst, color, 2);
			}

			if (undo) {
				undo = false;
				if(current_coord_vec.size() > 0) {
					full_image.copyTo(full_image_roi);
					current_coord_vec.pop_back();
				}
			}

            // marking is completed (left mouse button is OFF)
			if (selected)
			{
				selected = false;
				full_image.copyTo(full_image_roi);

				if (y_end < preview.rows && x_end > prev_img_rect.width && x_end < (full_image.cols - prev_img_rect.width) &&
					y_start < preview.rows)
				{
					int const i = (x_end - prev_img_rect.width) / preview.cols;
					trackbar_value += i;
				}
				else if (y_end >= preview.rows)
				{
					if (next_by_click) {
						++trackbar_value;
						current_coord_vec.clear();
					}

					Rect selected_rect(
						Point2i((int)min(x_start, x_end), (int)min(y_start, y_end)),
						Size(x_size, y_size));

					selected_rect &= full_rect_dst;
					selected_rect.y -= (int)prev_img_rect.height;

					coord_t coord;
					coord.abs_rect = selected_rect;
					coord.id = current_obj_id;
					current_coord_vec.push_back(coord);

					marks_changed = true;
				}
			}

			std::string current_synset_name;
			if (current_obj_id < synset_txt.size()) current_synset_name = "   - " + synset_txt[current_obj_id];

            // show X and Y coords of mouse
			if (show_mouse_coords) {
				full_image.copyTo(full_image_roi);
				int const x_inside = std::min((int)x_end, full_image_roi.cols);
				int const y_inside = std::min(std::max(0, y_end - (int)prev_img_rect.height), full_image_roi.rows);
				float const relative_center_x = (float)(x_inside) / full_image_roi.cols;
				float const relative_center_y = (float)(y_inside) / full_image_roi.rows;
				int const abs_x = relative_center_x*current_img_size.width;
				int const abs_y = relative_center_y*current_img_size.height;
				char buff[100];
				snprintf(buff, 100, "Abs: %d x %d    Rel: %.3f x %.3f", abs_x, abs_y, relative_center_x, relative_center_y);
				//putText(full_image_roi, buff, Point2i(800, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(50, 10, 10), 3);
				putText(full_image_roi, buff, Point2i(800, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 50, 50), 2);
				putText(full_image_roi, buff, Point2i(800, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(220, 120, 120), 1);
			}
			else
			{
				full_image.copyTo(full_image_roi);
				//std::string text = "Show mouse coordinates - press M";
				//putText(full_image_roi, text, Point2i(800, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 50, 50), 2);
				//putText(full_image_roi, text, Point2i(800, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(220, 120, 120), 1);
			}

            // marking is in progress (left mouse button is ON)
			if (draw_select)
			{
				if (add_id_img != 0) trackbar_value += add_id_img;

				if (y_start >= preview.rows)
				{
					//full_image.copyTo(full_image_roi);
					Rect selected_rect(
						Point2i(max(0, (int)min(x_start, x_end)), max(preview.rows, (int)min(y_start, y_end))),
						Point2i(max(x_start, x_end), max(y_start, y_end)));
					rectangle(frame, selected_rect, Scalar(150, 200, 150));

					if (show_mark_class)
					{
						putText(frame, std::to_string(current_obj_id) + current_synset_name,
							selected_rect.tl() + Point2i(2, 22), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(150, 200, 150), 2);
					}
				}
			}

			// Draw crosshair
			{
				const int offset = preview.rows; // Vertical offset

				// Only draw crosshair, if mouse is over image area
				if (y_end >= offset)
				{
					const bool bit_high = true;
					const bool bit_low = false;
					const int mouse_offset = 25;
					const int ver_min = draw_select ? std::min(x_end - mouse_offset, x_start - mouse_offset) : x_end - mouse_offset;
					const int ver_max = draw_select ? std::max(x_end + mouse_offset, x_start + mouse_offset) : x_end + mouse_offset;
					const int hor_min = draw_select ? std::min(y_end - mouse_offset, y_start - mouse_offset) : y_end - mouse_offset;
					const int hor_max = draw_select ? std::max(y_end + mouse_offset, y_start + mouse_offset) : y_end + mouse_offset;

					// Draw crosshair onto empty canvas (draws high bits on low-bit-canvas)
                    cv::Mat crosshair_mask(frame.size(), CV_8UC1, cv::Scalar(bit_low));
					cv::line(crosshair_mask, cv::Point(0, y_end), cv::Point(ver_min, y_end), cv::Scalar(bit_high)); // Horizontal, left to mouse
					cv::line(crosshair_mask, cv::Point(ver_max, y_end), cv::Point(crosshair_mask.size().width, y_end), cv::Scalar(bit_high)); // Horizontal, mouse to right
					cv::line(crosshair_mask, cv::Point(x_end, offset), cv::Point(x_end, std::max(offset, hor_min)), cv::Scalar(bit_high)); // Vertical, top to mouse
					cv::line(crosshair_mask, cv::Point(x_end, hor_max), cv::Point(x_end, crosshair_mask.size().height), cv::Scalar(bit_high)); // Vertical, mouse to bottom

					// Draw crosshair onto frame copy
					cv::Mat crosshair_frame(frame.size(), frame.type());
					frame.copyTo(crosshair_frame);
					cv::bitwise_not(crosshair_frame, crosshair_frame, crosshair_mask);

					// Fade-in frame copy with crosshair into original frame (for alpha)
					const double alpha = 0.7;
					cv::addWeighted(crosshair_frame, alpha, frame, 1 - alpha, 0.0, frame);
				}
			}

            // remove all labels from this image
			if (clear_marks == true)
			{
				clear_marks = false;
				marks_changed = true;
				full_image.copyTo(full_image_roi);
				current_coord_vec.clear();
			}


			if (old_current_obj_id != current_obj_id)
			{
				full_image.copyTo(full_image_roi);
				old_current_obj_id = current_obj_id;
				setTrackbarPos(trackbar_name_2, window_name, current_obj_id);
			}

            int selected_id = -1;
            // draw all labels
			//for (auto &i : current_coord_vec)
            for(size_t k = 0; k < current_coord_vec.size(); ++k)
			{
                auto &i = current_coord_vec.at(k);
				std::string synset_name;
				if (i.id < synset_txt.size()) synset_name = " - " + synset_txt[i.id];

				int offset = i.id * 25;
				int red = (offset + 0) % 255 * ((i.id + 2) % 3);
				int green = (offset + 70) % 255 * ((i.id + 1) % 3);
				int blue = (offset + 140) % 255 * ((i.id + 0) % 3);
				Scalar color_rect(red, green, blue);    // Scalar color_rect(100, 200, 100);

                // selected rect
                if (i.abs_rect.x < x_end && (i.abs_rect.x + i.abs_rect.width) > x_end &&
                    (i.abs_rect.y + preview.rows) < y_end && (i.abs_rect.y + i.abs_rect.height + preview.rows) > y_end)
                {
                    if (selected_id < 0) {
                        color_rect = Scalar(100, 200, 300);
                        selected_id = k;
                        rectangle(full_image_roi, i.abs_rect, color_rect, mark_line_width*2);
                    }
                }

				if (show_mark_class)
				{
					putText(full_image_roi, std::to_string(i.id) + synset_name,
						i.abs_rect.tl() + Point2f(2, 22), FONT_HERSHEY_SIMPLEX, 0.8, color_rect, 2);
				}

				rectangle(full_image_roi, i.abs_rect, color_rect, mark_line_width);
			}

            // remove selected rect
            if (delete_selected) {
                delete_selected = false;
                if (selected_id >= 0) current_coord_vec.erase(current_coord_vec.begin() + selected_id);
            }

            // show moving rect
            if (right_button_click == true)
            {
                if (move_rect_id < 0) move_rect_id = selected_id;

                int x_delta = x_end - x_start;
                int y_delta = y_end - y_start;
                auto rect = current_coord_vec[move_rect_id].abs_rect;
                rect.x += x_delta;
                rect.y += y_delta;

                Scalar color_rect = Scalar(300, 200, 100);
                rectangle(full_image_roi, rect, color_rect, mark_line_width);
            }

            // complete moving label rect
            if (move_rect && move_rect_id >= 0) {
                int x_delta = x_end - x_start;
                int y_delta = y_end - y_start;
                current_coord_vec[move_rect_id].abs_rect.x += x_delta;
                current_coord_vec[move_rect_id].abs_rect.y += y_delta;
                move_rect = false;
                move_rect_id = -1;
            }


            if (next_by_click) {
                putText(full_image_roi, "Mode: 1 mark per image (next by click)",
                    Point2i(850, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 170, 100), 2);
            }


			{
				std::string const obj_str = "Object id: " + std::to_string(current_obj_id) + current_synset_name;

				putText(full_image_roi, obj_str, Point2i(0, 21), FONT_HERSHEY_DUPLEX, 0.8, Scalar(10, 50, 10), 3);
				putText(full_image_roi, obj_str, Point2i(0, 21), FONT_HERSHEY_DUPLEX, 0.8, Scalar(20, 120, 60), 2);
				putText(full_image_roi, obj_str, Point2i(0, 21), FONT_HERSHEY_DUPLEX, 0.8, Scalar(50, 200, 100), 1);
			}

			if (show_help)
			{
				putText(full_image_roi,
					"<- prev_img    -> next_img    c - clear_marks    n - one_object_per_img    0-9 - obj_id    m - show coords    ESC - exit",
					Point2i(0, 45), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 10, 10), 2);
				putText(full_image_roi,
					"w - line width   k - hide obj_name   p - copy previous   o - track objects   r - delete selected   R-mouse - move box", //   h - disable help",
					Point2i(0, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 10, 10), 2);
			}
			else
			{
				putText(full_image_roi,
					"h - show help",
					Point2i(0, 45), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 10, 10), 2);
			}


			// arrows
			{
				Scalar prev_arrow_color(200, 150, 100);
				Scalar next_arrow_color = prev_arrow_color;
				if (prev_img_rect.contains(Point2i(x_end, y_end))) prev_arrow_color = Scalar(220, 190, 170);
				if (next_img_rect.contains(Point2i(x_end, y_end))) next_arrow_color = Scalar(220, 190, 170);

				std::vector<Point> prev_triangle_pts = { Point(5, 50), Point(40, 90), Point(40, 10), Point(5, 50) };
				Mat prev_roi = frame(prev_img_rect);
				line(prev_roi, prev_triangle_pts[0], prev_triangle_pts[1], prev_arrow_color, 5);
				line(prev_roi, prev_triangle_pts[1], prev_triangle_pts[2], prev_arrow_color, 5);
				line(prev_roi, prev_triangle_pts[2], prev_triangle_pts[3], prev_arrow_color, 5);
				line(prev_roi, prev_triangle_pts[3], prev_triangle_pts[0], prev_arrow_color, 5);

				std::vector<Point> next_triangle_pts = { Point(10, 10), Point(10, 90), Point(45, 50), Point(10, 10) };
				Mat next_roi = frame(next_img_rect);
				line(next_roi, next_triangle_pts[0], next_triangle_pts[1], next_arrow_color, 5);
				line(next_roi, next_triangle_pts[1], next_triangle_pts[2], next_arrow_color, 5);
				line(next_roi, next_triangle_pts[2], next_triangle_pts[3], next_arrow_color, 5);
				line(next_roi, next_triangle_pts[3], next_triangle_pts[0], next_arrow_color, 5);
			}

			imshow(window_name, frame);

#ifndef CV_VERSION_EPOCH
			int pressed_key = cv::waitKeyEx(20);	// OpenCV 3.x
#else
			int pressed_key = cv::waitKey(20);		// OpenCV 2.x
#endif

			if (pressed_key >= 0)
				for (int i = 0; i < 5; ++i) cv::waitKey(1);

			if (exit_flag) break;	// exit after saving
			if (pressed_key == 27 || pressed_key == 1048603) exit_flag = true;// break;  // ESC - save & exit

			if (pressed_key >= '0' && pressed_key <= '9') current_obj_id = pressed_key - '0';   // 0 - 9
			if (pressed_key >= 1048624 && pressed_key <= 1048633) current_obj_id = pressed_key - 1048624;   // 0 - 9

			switch (pressed_key)
			{
			//case 'z':		// z
			//case 1048698:	// z
			//    undo = true;
			//	break;

            case 'p':       // p
            case 1048688:	// p
                copy_previous_marks = 1;
                ++trackbar_value;
                break;

            case 'o':       // o
            case 1048687:	// o
                tracker_copy_previous_marks = 1;
                ++trackbar_value;
                break;

			case 32:        // SPACE
			case 1048608:	// SPACE
				++trackbar_value;
				break;

			case 2424832:   // <-
			case 65361:     // <-
			case 91:		// [
				--trackbar_value;
				break;
			case 2555904:   // ->
			case 65363:     // ->
			case 93:		// ]
				++trackbar_value;
				break;
			case 'c':       // c
			case 1048675:	// c
				clear_marks = true;
				break;
			case 'm':		// m
			case 1048685:   // m
				show_mouse_coords = !show_mouse_coords;
				full_image.copyTo(full_image_roi);
				break;
			case 'n':       // n
			case 1048686:   // n
				next_by_click = !next_by_click;
				full_image.copyTo(full_image_roi);
				break;
			case 'w':       // w
			case 1048695:   // w
				mark_line_width = mark_line_width % MAX_MARK_LINE_WIDTH + 1;
			break;
			case 'h':		// h
			case 1048680:	// h
				show_help = !show_help;
			break;
			case 'k':
			case 1048683:
				show_mark_class = !show_mark_class;
				break;
            case 'r':       // r
            case 1048690:   // r
                delete_selected = true;
                break;
			default:
				;
			}

            if(tracker_copy_previous_marks)
                tracker_optflow.update_tracking_flow(optflow_img, current_coord_vec);

			//if (pressed_key >= 0) std::cout << "pressed_key = " << (int)pressed_key << std::endl;

		} while (true);

	}
	catch (std::exception &e) {
		std::cout << "exception: " << e.what() << std::endl;
	}
	catch (...) {
		std::cout << "unknown exception \n";
	}

    return 0;
}
