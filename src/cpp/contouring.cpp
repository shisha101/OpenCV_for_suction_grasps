#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "opencv2/features2d/features2d.hpp" // for blob detector
//#include <omp.h> // for time computation
/**
 * Ideas: find a good way to merge intersecting and internal contours that way you get the majour contours without problems
 * Think about using key points for object recognition in parallel with the bounding box
 **/

using namespace cv;
using namespace std;

Mat src;
Mat src_gray;
//int thresh = 40;
//int max_thresh = 5000;
//int max_thresh_max = 5000;
//int thresh_max = 160;
int kernal_size = 1;
int gaussian_size = 1;
int area_thresh_hold = 900;
int area_thresh_hold_filtration_min = 150;
//int contour_smoothnes = 3;
int kernal_size_canny = 1;
RNG rng(12345);

/// Function header
void thresh_callback(int, void*);
void trav_tree(vector<Vec4i> hierarchy, int start_leaf,
		vector<RotatedRect> rectangles, vector<int>& result);

/** @function main */
int main(int argc, char** argv) {
	if (argc < 2) {
		src = imread("Query_images/13.jpg", 1);
	} else {
		src = imread(argv[1], 1);
	}
	/// Load source image and convert it to gray
//  src = imread( "Query_images/13.jpg", 1 );
	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_BGR2GRAY);
//  blur( src_gray, src_gray, Size(3,3) );

	/// Create Window
	string source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

//	createTrackbar(" Threshold:", "Source", &thresh, max_thresh,
//			thresh_callback);
//	createTrackbar(" Threshold Max:", "Source", &thresh_max, max_thresh_max,
//			thresh_callback);
	createTrackbar(" Gausian size", "Source", &gaussian_size, 9,
				thresh_callback);
	createTrackbar(" closing Kernal size", "Source", &kernal_size, 9, thresh_callback);
	createTrackbar(" canny Kernal size", "Source", &kernal_size_canny, 9, thresh_callback);
	createTrackbar(" Area_threshold", "Source", &area_thresh_hold, 50000,
			thresh_callback);
	createTrackbar(" Area_threshold_for_filteration", "Source", &area_thresh_hold_filtration_min, 50000,
				thresh_callback);
//	createTrackbar(" contour smoothenes", "Source", &contour_smoothnes, 100,
//			thresh_callback);
	thresh_callback(0, 0);

	waitKey(0);
	return (0);
}

/** @function thresh_callback */
void thresh_callback(int, void*) {

	Mat src_blured,src_copy_1,src_copy_2, src_copy_3;
	Mat threshold_output, gradient_estimate;
	Mat test_2,test_3, test_1;
	vector<vector<Point> > contours;
	vector<Vec4i> lines;
	vector<Vec4i> hierarchy;
	vector<int> possible_contours;
	vector<int> tree_heads;

	namedWindow("Image_after_blur", CV_WINDOW_AUTOSIZE);
	namedWindow("Edge detection", CV_WINDOW_AUTOSIZE);
	namedWindow("Closing of edges", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_original", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_smooth", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_smooth_on_image", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_smooth_filtered_on_image", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_boxes_on_image", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_filtered", CV_WINDOW_AUTOSIZE);
	namedWindow("gradient estimate", CV_WINDOW_AUTOSIZE);
	namedWindow("gradient estimate_treshold", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_smooth_filtered", CV_WINDOW_AUTOSIZE);
	namedWindow("Test", CV_WINDOW_AUTOSIZE);
//	namedWindow("Contours_filtered_area_final", CV_WINDOW_AUTOSIZE);
	src.copyTo(src_copy_1);
	src.copyTo(src_copy_2);
	src.copyTo(src_copy_3);

	GaussianBlur(src, src_blured,
			Size(gaussian_size * 2 + 1, gaussian_size * 2 + 1), 0);
	// blur the image
	GaussianBlur(src_gray, threshold_output,
			Size(gaussian_size * 2 + 1, gaussian_size * 2 + 1), 0);
	// extract gradient
	morphologyEx(threshold_output, gradient_estimate, CV_MOP_GRADIENT, getStructuringElement(0,
			Size(2 * kernal_size + 1, 2 * kernal_size + 1),
			Point(kernal_size, kernal_size)));
	imshow("gradient estimate", gradient_estimate);
	// threshold gradient
	threshold(gradient_estimate,gradient_estimate,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
	imshow("gradient estimate_treshold", gradient_estimate);
	// extract the edges from gradient
	double CannyAccThresh = threshold(threshold_output,test_3,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
	Canny(threshold_output, threshold_output, 0.35*CannyAccThresh, CannyAccThresh, 2*kernal_size_canny+1, true);
	// display results of edge detection
	imshow("Edge detection", threshold_output);
	// Trial code
//
//	SimpleBlobDetector::Params params;
//	vector<KeyPoint> keypoints;
//	vector<vector<Point> > contours_bobs;
//	Mat im_with_keypoints;
//	params.minThreshold = 0;//0.35*CannyAccThresh;
//	params.maxThreshold = 180;//CannyAccThresh;
//	params.filterByArea = true;
//	params.minArea = area_thresh_hold_filtration_min;
//	params.minDistBetweenBlobs = 4;
//	SimpleBlobDetector detector(params);
//	detector.detect( src_gray, keypoints);
//	drawKeypoints( src_gray, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
//	for (int i = 0;i<keypoints.size(); i++){
//	drawContours( im_with_keypoints, keypoints[i]., i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
//	}
//	imshow("Test", im_with_keypoints);


	// end of trail code

	//fill in discontinuities
	Mat element = getStructuringElement(0,
			Size(2 * kernal_size + 1, 2 * kernal_size + 1),
			Point(kernal_size, kernal_size));
	morphologyEx(threshold_output, threshold_output, CV_MOP_CLOSE, element);
	imshow("Closing of edges", threshold_output);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE,
			CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Find the rotated rectangles and ellipses for each contour
	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());
	vector<vector<Point> > contours_smooth(contours.size());

// fit rectangles
	cout<<"Nxt, Prv, FC, Par \n";
	for (int i = 0; i < contours.size(); i++) {
		minRect[i] = minAreaRect(Mat(contours[i]));
		if (hierarchy[i][3] == -1
				&& minRect[i].size.width * minRect[i].size.height
						>= area_thresh_hold) {
			tree_heads.push_back(i);
		}
//		double eps = (contour_smoothnes/200.0)*arcLength(contours[i],true);
//		approxPolyDP(contours[i],contours_smooth[i],eps,true);
		convexHull(contours[i],contours_smooth[i]);

		cout<< i <<"->"<<hierarchy[i]<<"-->"<< (isContourConvex(contours[i])) <<"-->"<< (minRect[i].size.width * minRect[i].size.height >= area_thresh_hold)<<"\n";
//       cout<<minRect[i].size.width<<"  "<< minRect[i].size.height<<"\n";
	}
	cout<<"Tree heads: "<<"\n";
	for(int i=0; i<tree_heads.size();i++)
	{
		cout<<tree_heads[i]<< ", ";
	}
	cout<< " \nend of tree heads \n";

	// add the recursion here use the result in the following loop and plot at the end
	// iterate on tree heads
	for (int i = 0; i < tree_heads.size(); i++) {
		trav_tree(hierarchy, tree_heads[i], minRect, possible_contours);
	}

	/// Draw contours + rotated rects + ellipses
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	Mat drawing_boxes = Mat::zeros(threshold_output.size(), CV_8UC3);
	Mat drawing_smooth = Mat::zeros(threshold_output.size(), CV_8UC3);
	Mat drawing_smooth_filtered = Mat::zeros(threshold_output.size(), CV_8UC3);
	Mat drawing_filterd = Mat::zeros(threshold_output.size(), CV_8UC3);

	for (int i = 0; i < contours.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		// contour
       drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       drawContours( drawing_smooth, contours_smooth, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       drawContours( src_copy_2, contours_smooth, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       if(contourArea(contours_smooth[i]) >= area_thresh_hold_filtration_min){
    	   drawContours( drawing_smooth_filtered, contours_smooth, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
   		Point2f rect_points[4];
   		minRect[i].points(rect_points);
   		for (int j = 0; j < 4; j++)
   			line(drawing_boxes, rect_points[j], rect_points[(j + 1) % 4], color, 1,
   					8);
       }
		// ellipse
//       ellipse( drawing, minEllipse[i], color, 2, 8 );
		// rotated rectangle
//		Point2f rect_points[4];
//		minRect[i].points(rect_points);
//		for (int j = 0; j < 4; j++)
//			line(drawing_boxes, rect_points[j], rect_points[(j + 1) % 4], color, 1,
//					8);
//       waitKey(3000);
	}


	for (int i = 0; i < possible_contours.size(); i++) {
		cout<<possible_contours[i]<<", ";
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		Point2f rect_points[4];
		minRect[possible_contours[i]].points(rect_points);
		drawContours( src_copy_3, contours_smooth, possible_contours[i], color, 2, 8, vector<Vec4i>(), 0, Point() );
		for (int j = 0; j < 4; j++){
			line(drawing_filterd, rect_points[j], rect_points[(j + 1) % 4],
					color, 1, 8);
			line(src_copy_1, rect_points[j], rect_points[(j + 1) % 4],
								Scalar(0, 255, 0), 2, 8);
		}
	}
	/// Show in a window
//  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imshow("Contours_filtered", drawing_filterd);
	imshow("Contours_original", drawing);
	imshow("Contours_smooth", drawing_smooth);
	imshow("Contours_smooth_filtered", drawing_smooth_filtered);
	imshow("Image_after_blur", src_blured);
	imshow("Contours_smooth_on_image",src_copy_2);
	imshow("Contours_boxes_on_image",src_copy_1);
	imshow("Contours_smooth_filtered_on_image",src_copy_3);
	imshow("Test",drawing_boxes);
}

void trav_tree(vector<Vec4i> hierarchy, int start_leaf,
		vector<RotatedRect> rectangles, vector<int>& result) {

	vector<int> children;
	cout<<"the parent leaf is: "<< start_leaf<<"\n";
	// extract all children of the current leaf
	if (hierarchy[start_leaf][2] != -1)       // children exist
			{
		int child_index = hierarchy[start_leaf][2];
		children.push_back(child_index);
		cout<<"the children are : \n"<< children.back();
		int next_child_index = hierarchy[child_index][0];

		while (next_child_index != -1)       // get all children
		{
			children.push_back(next_child_index);
			next_child_index = hierarchy[next_child_index][0];    // find next peer
			cout<<", "<< children.back();
		}
		cout<<"\n";

		bool some_large = false;
		for (int i = 0; i < children.size(); i++) {
			if (rectangles[children[i]].size.width
					* rectangles[children[i]].size.height >= area_thresh_hold) // if chiled is large enough explore
					{
				some_large = true;
				trav_tree(hierarchy, children[i], rectangles, result);
			}
		}
		if (some_large == false) {// no child is larger than area threshold
			result.push_back(start_leaf);
			cout<<"no child is larger than threshold \n";
			cout<<"The last result is: "<< result.back() <<"\n";

		}

	}
	else // has no children
	{
		if (rectangles[start_leaf].size.width
				* rectangles[start_leaf].size.height >= area_thresh_hold) // if large enough append else ignore
				{
			result.push_back(start_leaf);
			cout<<"no child exists \n";
			cout<<"The last result is: "<< result.back() <<"\n";
		}
	}

}

/*
 * 	HoughLinesP(threshold_output,lines,5, 3*CV_PI/180, 80, 70, 10);
		for( size_t i = 0; i < lines.size(); i++ )
		    {
		        line( src_copy_3, Point(lines[i][0], lines[i][1]),
		            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 1, 8 );
		    }
		imshow("Lines", src_copy_3);


		// atempt

	Mat threshold_output_filterd,threshold_output_filterd2, plot_filtered;
	vector<vector<Point> > contours_filtered;
	vector<Vec4i> hierarchy_filtered;
	for(int i=0;i<contours.size();i++)
	{
		if (contourArea(contours[i])>=area_thresh_hold_filtration){
			contours_filtered.push_back(contours[i]);
		}
	}
	Mat draw_filtered_contours = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i=0;i<contours_filtered.size();i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
						rng.uniform(0, 255));
		drawContours( draw_filtered_contours, contours_filtered, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	}
	morphologyEx(draw_filtered_contours, draw_filtered_contours, CV_MOP_CLOSE, element);
//	GaussianBlur(draw_filtered_contours, draw_filtered_contours,
//				Size(gaussian_size * 2 + 1, gaussian_size * 2 + 1), 0);
	imshow("Contours_filtered_area", draw_filtered_contours);
	cvtColor(draw_filtered_contours, threshold_output_filterd2, CV_BGR2GRAY);
	findContours(threshold_output_filterd2, contours_filtered, hierarchy_filtered, CV_RETR_TREE,
				CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat draw_filtered_contours2 = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i=0;i<contours_filtered.size();i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
							rng.uniform(0, 255));
			drawContours( draw_filtered_contours2, contours_filtered, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		}
	imshow("Contours_filtered_area_final",draw_filtered_contours2);
 */

