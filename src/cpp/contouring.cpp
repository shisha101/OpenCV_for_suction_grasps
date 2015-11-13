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
//#define NDEBUG

vector<Mat> src;
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
int current_index = 4;
string image_directory = "resources/Query_images/";
string image_extention = ".jpg";
bool debug = true;
double contour_smoothnes = 0.1; // parameter used for approxPolyDP
RNG rng(12345);

/// Function header
struct contour_container{

	vector<Point> contour_raw;
	vector<Point> contour_convex;
//	vector<Point> contour_concave; // does not exist in open CV check PCL
	vector<Point> contour_Poly_approx;
	RotatedRect   min_bb;
	Point2i 	  convex_hull_centeroid;
//	Point2i 	  concave_hull_centeroid; // does not exist in open CV check PCL
	Vec4i         hierarchy;
	string		  source_image;
	int 		  index_original;  // the index of the contour in the original list of contours
void calc_convex_hull_centeroid()
{
	Moments m_contours = moments(contour_convex);
	convex_hull_centeroid = Point2i( m_contours.m10/m_contours.m00 , m_contours.m01/m_contours.m00 );
}
};
void thresh_callback(int, void*);
void trav_tree(vector<Vec4i>& hierarchy, int start_leaf,
		vector<contour_container>& rectangles, vector<int>& result, vector<contour_container>& resulting_contours);

/** @function main */
int main(int argc, char** argv) {
	if (argc < 2) {
		src.push_back(imread(image_directory+"13.jpg", 1));
	} else if (argc == 2){
		src.push_back(imread(argv[1]+image_extention, 1));
	}
	else {
		for (int i = 1; i< argc; i++)
		{
			src.push_back(imread(image_directory+argv[i]+image_extention, 1));
			string temp = image_directory+argv[i];
		}
	}
	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	//	createTrackbar(" Threshold:", "Source", &thresh, max_thresh,
//			thresh_callback);
//	createTrackbar(" Threshold Max:", "Source", &thresh_max, max_thresh_max,
//			thresh_callback);
	createTrackbar("Gausian size", "Source", &gaussian_size, 9, thresh_callback);
	createTrackbar("closing Kernal size", "Source", &kernal_size, 9, thresh_callback);
	createTrackbar("canny Kernal size", "Source", &kernal_size_canny, 9, thresh_callback);
	createTrackbar("Area_threshold (affects tree and computations)", "Source", &area_thresh_hold, 50000, thresh_callback);
	createTrackbar("Image inex", "Source", &current_index, src.size(),	thresh_callback);
	createTrackbar("Area_threshold_for_plotting (affects plotting raw contours (smooth))", "Source",
			&area_thresh_hold_filtration_min, 50000, thresh_callback);
//	createTrackbar(" contour smoothenes", "Source", &contour_smoothnes, 100,
//			thresh_callback);
#ifdef NDEBUG
	// debug windows
		namedWindow("Contours_original", CV_WINDOW_AUTOSIZE);
		namedWindow("Contours_smooth", CV_WINDOW_AUTOSIZE);
		namedWindow("Contours_smooth_on_image", CV_WINDOW_AUTOSIZE);
		namedWindow("Test", CV_WINDOW_AUTOSIZE);
		namedWindow("gradient estimate", CV_WINDOW_AUTOSIZE);
		namedWindow("Edge detection", CV_WINDOW_AUTOSIZE);
		namedWindow("gradient estimate_treshold", CV_WINDOW_AUTOSIZE);
#endif
	// permanent windows

//	namedWindow("Image_after_blur", CV_WINDOW_AUTOSIZE);
//	namedWindow("Closing of edges", CV_WINDOW_AUTOSIZE);

	namedWindow("Contours_filtered", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_smooth_filtered", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_boxes_on_image", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours_smooth_filtered_on_image", CV_WINDOW_AUTOSIZE);
//	namedWindow("Contours_filtered_area_final", CV_WINDOW_AUTOSIZE);

	thresh_callback(0, 0);

	waitKey(0);
	return (0);
}

/** @function thresh_callback */
void thresh_callback(int, void*) {

	Mat src_blured,src_copy_1,src_copy_2, src_copy_3;
	Mat threshold_output, gradient_estimate,gradient_estimate_thresh;
	Mat test_2,test_3, test_1;
	vector<vector<Point> > contours;
	vector<Vec4i> lines;
	vector<Vec4i> hierarchy;
	vector<int> possible_contours;
	vector<int> tree_heads;
	if (current_index >= src.size())
	{
		current_index --; // solves the problem of the limit cannot equal the initial value of the Trackbar
	}

	src[current_index].copyTo(src_copy_1);
	src[current_index].copyTo(src_copy_2);
	src[current_index].copyTo(src_copy_3);
	cvtColor(src[current_index], src_gray, CV_BGR2GRAY);
	GaussianBlur(src[current_index], src_blured,
			Size(gaussian_size * 2 + 1, gaussian_size * 2 + 1), 0);
	// blur the image
	GaussianBlur(src_gray, threshold_output,
			Size(gaussian_size * 2 + 1, gaussian_size * 2 + 1), 0);
	// extract gradient
	morphologyEx(threshold_output, gradient_estimate, CV_MOP_GRADIENT, getStructuringElement(0,
			Size(2 * kernal_size + 1, 2 * kernal_size + 1),
			Point(kernal_size, kernal_size)));
	// threshold gradient
	threshold(gradient_estimate,gradient_estimate_thresh,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);

	// extract the edges from gradient
	double CannyAccThresh = threshold(threshold_output,test_3,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU); // used for auto thresholding
	Canny(threshold_output, threshold_output, 0.35*CannyAccThresh, CannyAccThresh, 2*kernal_size_canny+1, true);
	// display results of edge detection
	// Trial code
//
//	SimpleBlobDetector::Params params;
//	vector<KeyPoint> keypoints;
//	vector<vector<Point> > contours_bobs;
//	Mat im_with_keypoints;
//	params.minThreshold = 0*CannyAccThresh;
//	params.maxThreshold = CannyAccThresh;
//	params.thresholdStep = 10;
//	params.filterByArea = true;
////	params.filterByColor = true;
////	params.blobColor = 200;
//	params.minArea = 1;
//	params.maxArea = 500000;
//	params.minDistBetweenBlobs = 0;
//	SimpleBlobDetector detector(params);
//	detector.detect( src_gray, keypoints);
//	cout<<"number of key points :"<< keypoints.size()<<"\n";
//	drawKeypoints( src_gray, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
////	for (int i = 0;i<keypoints.size(); i++){
////	drawContours( im_with_keypoints, keypoints[i]., i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
////	}
//	imshow("Test", im_with_keypoints);


	// end of trail code

	//fill in discontinuities
	Mat element = getStructuringElement(0,
			Size(2 * kernal_size + 1, 2 * kernal_size + 1),
			Point(kernal_size, kernal_size));
	morphologyEx(threshold_output, threshold_output, CV_MOP_CLOSE, element);
//	imshow("Closing of edges", threshold_output);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE,
			CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
///
// editing starts
///
//	Mat contours_printed = Mat::zeros(threshold_output.size(), CV_8UC1);
//	Mat find_contours_2 = Mat::zeros(threshold_output.size(), CV_8UC1);
//	Mat source_diff_color_space;
//	vector<vector<Point> > contours_smooth_trial(contours.size());
//	vector<vector<Point> > contours_smooth_trial_2;
//	vector<Vec4i> hierarchy_2;
//	vector<Mat> color_spaces;
//	cvtColor(src_blured, source_diff_color_space,COLOR_BGR2Lab);
//	split(source_diff_color_space,color_spaces);
//	namedWindow("color_space", CV_WINDOW_AUTOSIZE);
//	imshow("color_space",color_spaces[2]);
//	for (int i = 0; i<contours.size();i ++){
//	convexHull(contours[i],contours_smooth_trial[i]);
//	drawContours(contours_printed, contours_smooth_trial, i, Scalar(255));//, 1, 8, vector<Vec4i>(), 0, Point() );
//	}
//
//	findContours(color_spaces[2], contours_smooth_trial_2, hierarchy_2, CV_RETR_TREE,
//				CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//	for (int i = 0; i<contours_smooth_trial_2.size();i ++){
////		convexHull(contours[i],contours_smooth_trial[i]);
//		drawContours(find_contours_2, contours_smooth_trial_2, i, Scalar(255,255,255));//, 1, 8, vector<Vec4i>(), 0, Point() );
//		}
//
//    namedWindow("Contours_printed", CV_WINDOW_AUTOSIZE);
//	imshow("Contours_printed",contours_printed);
//	namedWindow("find_contours_trial", CV_WINDOW_AUTOSIZE);
//	imshow("find_contours_trial",find_contours_2);
///
// end of edditing
///


	/// Find the rotated rectangles and ellipses for each contour
	vector<contour_container> contour_cont_vect(contours.size());
	vector<contour_container> resulting_cont;
	vector<RotatedRect> minRect(contours.size());
//	vector<RotatedRect> minEllipse(contours.size());
	vector<vector<Point> > contours_smooth(contours.size());

// fit rectangles
#ifdef NDEBUG
	cout<<"indx, Nxt, Prv, FC, Par, convex?, >area thresh \n";
#endif
	for (int i = 0; i < contours.size(); i++) {
		contour_cont_vect[i].min_bb = minAreaRect(Mat(contours[i]));
		convexHull(contours[i],contour_cont_vect[i].contour_convex);
		contour_cont_vect[i].hierarchy = hierarchy[i];
		double eps = (contour_smoothnes)*arcLength(contours[i],true);
		approxPolyDP(contours[i],contour_cont_vect[i].contour_Poly_approx,eps,true);
		// if contour has no parent, it is a head, if a head is larger than the minimum size it is considered
		if (hierarchy[i][3] == -1
				&& contour_cont_vect[i].min_bb.size.area() >= area_thresh_hold) {
			tree_heads.push_back(i);
		}
#ifdef NDEBUG
		cout<< i <<"->"<<hierarchy[i]<<"-->"<< (isContourConvex(contours[i])) <<"-->"<< (contour_cont_vect[i].min_bb.size.area() >= area_thresh_hold)<<"\n";
#endif
		minRect[i] = contour_cont_vect[i].min_bb;
		contours_smooth[i] = contour_cont_vect[i].contour_convex;
	}
#ifdef NDEBUG
	cout<<"Tree heads: "<<"\n";

	for(int i=0; i<tree_heads.size();i++)
	{
		cout<<tree_heads[i]<< ", ";
	}
	cout<< " \nend of tree heads \n";
#endif
	// add the recursion here use the result in the following loop and plot at the end
	// iterate on tree heads
	for (int i = 0; i < tree_heads.size(); i++) {
		trav_tree(hierarchy, tree_heads[i], contour_cont_vect, possible_contours, resulting_cont);
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
       drawContours( drawing_smooth, vector<vector<Point> > (1,contour_cont_vect[i].contour_convex), 0, color, 1, 8, vector<Vec4i>(), 0, Point() );
       drawContours( src_copy_2, vector<vector<Point> > (1,contour_cont_vect[i].contour_convex), 0, color, 1, 8, vector<Vec4i>(), 0, Point() );
       if(contourArea(contour_cont_vect[i].contour_convex) >= area_thresh_hold_filtration_min){
    	   drawContours( drawing_smooth_filtered, vector<vector<Point> > (1,contour_cont_vect[i].contour_convex), 0, color, 1, 8, vector<Vec4i>(), 0, Point() );
   		Point2f rect_points[4];
   		contour_cont_vect[i].min_bb.points(rect_points);
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


//	for (int i = 0; i < possible_contours.size(); i++) {
//		cout<<possible_contours[i]<<", ";
//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
//				rng.uniform(0, 255));
//		Point2f rect_points[4];
//		minRect[possible_contours[i]].points(rect_points);
//		drawContours( src_copy_3, contours_smooth, possible_contours[i], color, 2, 8, vector<Vec4i>(), 0, Point() );
//		Moments m_contours = moments(contours_smooth[possible_contours[i]]);
//		Point2i centeroid = Point2i( m_contours.m10/m_contours.m00 , m_contours.m01/m_contours.m00 );
//		circle(src_copy_3, centeroid,5,Scalar(0,0,255),-1);
//		circle(drawing_filterd,minRect[possible_contours[i]].center,5,Scalar(0,0,255),-1);
//		circle(src_copy_1,minRect[possible_contours[i]].center,5,Scalar(0,0,255),-1);
//		for (int j = 0; j < 4; j++){
//			line(drawing_filterd, rect_points[j], rect_points[(j + 1) % 4],
//					color, 1, 8);
//			line(src_copy_1, rect_points[j], rect_points[(j + 1) % 4],
//								Scalar(0, 255, 0), 2, 8);
//		}
//	}
//

	for (vector<contour_container>::iterator CC = resulting_cont.begin(); CC != resulting_cont.end(); CC++) {
//		cout<<possible_contours[i]<<", ";
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		Point2f rect_points[4];
		CC->min_bb.points(rect_points);
		drawContours( src_copy_3, vector<vector<Point> > (1,CC->contour_convex), 0,  color, 2, 8, vector<Vec4i>(), 0, Point() );
		CC->calc_convex_hull_centeroid();
		circle(src_copy_3, CC->convex_hull_centeroid,5,Scalar(0,0,255),-1);
		circle(drawing_filterd,CC->min_bb.center,5,Scalar(0,0,255),-1);
		circle(src_copy_1,CC->min_bb.center,5,Scalar(0,0,255),-1);
		for (int j = 0; j < 4; j++){
			line(drawing_filterd, rect_points[j], rect_points[(j + 1) % 4],
					color, 1, 8);
			line(src_copy_1, rect_points[j], rect_points[(j + 1) % 4],
								Scalar(0, 255, 0), 2, 8);
		}
	}

	/// Show in a window
//  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	// debug windows
#ifdef NDEBUG
	imshow("Contours_original", drawing);
	imshow("Contours_smooth", drawing_smooth);
	imshow("Contours_smooth_on_image",src_copy_2);
	imshow("Test",drawing_boxes);
	imshow("gradient estimate", gradient_estimate);
	imshow("Edge detection", threshold_output);
	imshow("Image_after_blur", src_blured);
	imshow("gradient estimate_treshold", gradient_estimate_thresh);
#endif
	// permanent windows
	imshow("Contours_smooth_filtered", drawing_smooth_filtered);
	imshow("Contours_filtered", drawing_filterd);
	imshow("Contours_boxes_on_image",src_copy_1);
	imshow("Contours_smooth_filtered_on_image",src_copy_3);
	imshow("Source", src[current_index]);


}

void trav_tree(vector<Vec4i>& hierarchy, int start_leaf,
		vector<contour_container>& contours_in_tree, vector<int>& result, vector<contour_container>& resulting_contours) {

	vector<int> children;
#ifdef NDEBUG
	cout<<"the parent leaf is: "<< start_leaf<<"\n";
#endif
	// extract all children of the current leaf
	if (hierarchy[start_leaf][2] != -1)       // children exist
			{
		int child_index = hierarchy[start_leaf][2];
		children.push_back(child_index);
#ifdef NDEBUG
		cout<<"the children are : \n"<< children.back();
#endif
		int next_child_index = hierarchy[child_index][0];

		while (next_child_index != -1)       // get all children
		{
			children.push_back(next_child_index);
			next_child_index = hierarchy[next_child_index][0];    // find next peer
#ifdef NDEBUG
			cout<<", "<< children.back();
#endif
		}
#ifdef NDEBUG
		cout<<"\n";
#endif
		bool some_large = false;
		for (int i = 0; i < children.size(); i++) {
			if (contours_in_tree[children[i]].min_bb.size.area() >= area_thresh_hold) // if chiled is large enough explore
					{
				some_large = true;
				trav_tree(hierarchy, children[i], contours_in_tree, result, resulting_contours);
			}
		}
		if (some_large == false) {// no child is larger than area threshold
			result.push_back(start_leaf);
			contours_in_tree[start_leaf].index_original = start_leaf;
			resulting_contours.push_back(contours_in_tree[start_leaf]);
#ifdef NDEBUG
			cout<<"no child is larger than threshold \n";
			cout<<"The last result is: "<< result.back() <<"\n";
#endif
		}

	}
	else // has no children
	{
		if (contours_in_tree[start_leaf].min_bb.size.area() >= area_thresh_hold) // if large enough append else ignore
				{
			result.push_back(start_leaf);
			contours_in_tree[start_leaf].index_original = start_leaf;
			resulting_contours.push_back(contours_in_tree[start_leaf]);
#ifdef NDEBUG
			cout<<"no child exists \n";
			cout<<"The last result is: "<< result.back() <<"\n";
#endif
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

