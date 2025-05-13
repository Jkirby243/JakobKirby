#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui/highgui_c.h>
//#include <opencv2/xphoto/inpainting.hpp>
#include <iostream>
#include <deque>

using namespace cv;
using namespace std;

int main() {

	//Preprecessing
	//For Manual change these to the unpait and Grount truth associated

	cv::Mat img = cv::imread("../OpenCVImages/Inpaints/InpaintAC0.png", IMREAD_GRAYSCALE);
	cv::Mat origimg = cv::imread("../OpenCVImages/Inpaints/InpaintAC0.png");
	cv::Mat GT = cv::imread("../OpenCVImages/Inpaints/GGT6.jpg", IMREAD_GRAYSCALE);; //Ground truth

	//Section For Inpainting

	//Method for reading a bunch of images; Doing just a large chunck of inpainting and black hatting so I can just run it on all the images
		//Creating BlackHatFilter
		//Uncomment from the line below to the end of the comment block for preprocessing
		//cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
		//Point filtersize = Point(14, 14);
		//Mat kernal = getStructuringElement(MORPH_RECT, filtersize);
		//vector<cv::String> fn;
		//glob("../OpenCVImages/Upload_20230214-235509~/Acne/*.jpg", fn, false);

		//vector<Mat> images;
		//size_t count = fn.size(); //number of png files in images folder
		//for (size_t i = 0; i < count; i++)
		//	images.push_back(imread(fn[i]));
		//int i = 0;
		//for (Mat img : images) { //227
			//Mat eq;
			//Mat origimg = img;
			//cvtColor(img, img, COLOR_BGR2GRAY);
			//equalizeHist(img, img);
			//Mat blackhatimg, mask;
			//Run blackhat filter
			//cv::morphologyEx(img, blackhatimg, MORPH_BLACKHAT, kernal);
			//create mask by binary threshholding using the blackhat
			//cv::threshold(blackhatimg, mask, 12, 255, THRESH_BINARY);
			//std::cout << "Black Hatting done" << endl;
			//namedWindow("Blackhat", WINDOW_AUTOSIZE);
			/*cv::imshow("Blackhat", mask);
			cv::moveWindow("Blackhat", 0, 90);
			cv::waitKey(0);*/
			/*for (int x = 0; x < img.cols; x++)
			{
				for (int y = 0; y < img.rows; y++)
				{
					if (img.at<Vec3f>(y, x) > 180)
					{
						img.at<Vec3f>(y, x);
					}
				}
			}*/
			//waitKey(0);
			//std::cout << "Doing inpainting" << endl;
			//Mat inpaint;
			//cv::inpaint(origimg, mask, inpaint, 40, INPAINT_TELEA);
			//cv::Mat origimg = cv::imread("../OpenCVImages/Inpaint.png");

			//cout << "Writing" << endl;
			//imwrite("../OpenCVImages/Inpaints/InpaintAC" + to_string(i) + ".png", inpaint);
			//Mat compare;
			//imwrite("../OpenCVImages/Inpaints/Compares/Compare" + to_string(i) + ".png", compare);
			//++i;
		//}

	//For Manual Comment Out this block--------------------
	vector<cv::String> fn;
	glob("../OpenCVImages/Inpaints/*.png", fn, false);
	vector<Mat> images;
	size_t count = fn.size(); //number of png files in images folder
	for (size_t i = 0; i < count; i++)
		images.push_back(imread(fn[i]));
	vector<cv::String> fn2;
	glob("../OpenCVImages/Inpaints/*.jpg", fn2, false);

	vector<Mat> GTS;
	count = fn2.size(); //number of png files in images folder
	for (size_t i = 0; i < count; i++){
		GTS.push_back(imread(fn2[i]));
		}
		int CTR = 0;
		for (Mat img : images) { //227
			Mat origimg = img.clone();
			GT = GTS[CTR];
			//To here --------------------------------------


			cvtColor(img, img, cv::COLOR_BGR2GRAY);
			Mat k = getGaussianKernel(8, 8);
			filter2D(origimg, origimg, 0, k, Point(-1, -1), 0.0, BORDER_REPLICATE);

			Mat Converted;
			//
			cvtColor(origimg, Converted, cv::COLOR_RGB2GRAY);


		
			//They used a modified otsu method. I will see if I can try and do a manual version of their method BUT
			//For now I will do the base otsu thresholding method
			Mat mask2;
			long double thres = threshold(Converted, mask2, 0, 255, THRESH_OTSU + THRESH_TRUNC);
			cout << "Otsu Threshold : " << thres << endl;
			thres = threshold(mask2, mask2, 0, 255, THRESH_BINARY + THRESH_OTSU);

			cout << "Otsu Threshold : " << thres << endl;
			//Fill hols in mask
			Point ksize = Point(14, 14);
			Mat Kern = getStructuringElement(MORPH_ELLIPSE, ksize);
			morphologyEx(mask2, mask2, MORPH_CLOSE, Kern);
			ksize = Point(4, 4);
			Mat flood = mask2.clone();
			floodFill(flood, Point(0, 0), Scalar(0));

			Mat tmp = mask2.clone();
			/*namedWindow("Mask2 preeflood", WINDOW_AUTOSIZE);
			cv::imshow("Mask2 preeflood", tmp);
			cv::moveWindow("Mask2 preeflood", 0, 90);*/

			bool equal = true;
			for (int x = 0; x < flood.cols; x++)
			{
				for (int y = 0; y < flood.rows; y++)
				{
					if (flood.at<uchar>(Point(x, y)) != mask2.at<uchar>(Point(x, y)))
					{
						cout << "NOT EQUAL AT " << Point(x, y) << endl;
						equal = false;
						break;
					}
				}
				if (equal == false) {
					break;
				}
			}
			if (equal == false) {
				mask2 = (mask2 | flood);
			}

			if (equal == false) {
				for (int x = 0; x < flood.cols; x++)
				{
					for (int y = 0; y < Converted.rows; y++)
					{
						//AT WORKED WHEN PUTTING IT IN A EMPTY MAT UNLIKE DATA! This will confuse me till the end of time as I have no clue what i sleepily did on tuesday
						if (flood.at<uchar>(Point(x, y)) == 255)
						{
							mask2.at<uchar>(Point(x, y)) = 0;
						}
					}
				}
			}
			/*namedWindow("flood", WINDOW_AUTOSIZE);
			cv::imshow("flood", flood);
			cv::moveWindow("flood", 0, 90);*/
			vector<tuple<Point, Point>> edges;
			//Fill in mask 2
			cout << "grabbing edges" << endl;
			for (int x = 0; x < mask2.rows; x++)
			{
				Point left, right = Point(-1, -1);
				//cout << "left = " << left << endl;
				for (int y = 0; y < mask2.cols; y++)
				{
					//AT WORKED WHEN PUTTING IT IN A EMPTY MAT UNLIKE DATA! This will confuse me till the end of time as I have no clue what i sleepily did on tuesday
					if (mask2.at<uchar>(Point(y, x)) == 0 && left == Point(0, 0))
					{
						left = Point(y, x);
					}
					if (mask2.at<uchar>(Point(y, x)) == 255 && y - 1 >= 0 && mask2.at<uchar>(Point(y - 1, x)) == 0 && left != Point(0, 0)) {
						right = Point(y - 1, x);
						edges.push_back(make_tuple(left, right));
						//cout << "left = " << left << " Right = " << right << endl;
						left = right = Point(0, 0);
					}
				}
			}

			int height = 0;
			int width = 0;

			for (auto tmp : edges)
			{
				width += (get<0>(tmp).y + get<1>(tmp).y) / 2;
				height += (get<0>(tmp).x + get<1>(tmp).x) / 2;
			}

			height = height / edges.size();
			width = width / edges.size();
			cout << "h = " << height << " w= " << width << endl;
			int x = width;
			int y = height;
			int hthresh, wthresh;
			Point tmph, tmpw;
			hthresh = wthresh = -1;
			//while(hthresh == -1) {
			//	//cout << "checking " << Point(mask2.cols - width, y) << endl;
			//	if (mask2.at<uchar>(Point(mask2.cols - width, y)) == 255 && mask2.at<uchar>(Point(mask2.cols - width, y - 1)) == 0) {
			//		hthresh =  y - 1;
			//	};
			//	++y;
			//}

			cout << "edge count" << edges.size() << endl;
			for (auto tmp : edges)
			{

				if (get<0>(tmp).y > hthresh) {
					//cout << get<0>(tmp) << " " << get<1>(tmp) << " " << (get<1>(tmp).x < 500) << endl;
				}
				if (get<1>(tmp).x >= mask2.cols - 10) {
					//cout << "width catch" << get<0>(tmp) << " " << get<1>(tmp) << " " << (get<1>(tmp).x < 500) << endl;
				}
				if (get<0>(tmp).x == 0 || get<1>(tmp).x < 500 || get<1>(tmp).y > 1020) {
					if (get<0>(tmp).y > hthresh) {
						//cout << "filling " << get<0>(tmp) << "-" << get<1>(tmp) << endl;
					}
					for (int x = get<0>(tmp).x; x <= get<1>(tmp).x; ++x) {
						//cout << "filled at " << Point(x, get<1>(tmp).y) << endl;
						mask2.at<uchar>(Point(x, get<1>(tmp).y)) = 255;
					}
				}

			}

			/*for (int x = 0; x < mask2.cols; ++x) {

				if (mask2.at<uchar>(Point(x,mask2.rows - 1)) == 0)
				{
					cout << Point(x, mask2.rows - 1) << endl;
				}
			}*/



			//	if (left.x == 100000000 || right.x == -100000000) {
			//		continue;
			//	}
			//	else {
			//		edges.push_back(make_tuple(left, right));
			//	}
			//}
			//for (auto t : edges) {
			//	cout << get<0>(t) << " " << get<1>(t) << endl;
			//}
			//
			////cout << "Doing edge fill" << endl;
			//for (auto t : edges) {
			//	Point left = get<0>(t);
			//	Point right = get<1>(t);
			//	int x = left.x;
			//	//cout << "filling " << left << "-" << right << endl;
			//	for (int y = left.y; y <= right.y; ++y) {
			//		//cout << "filling " << left << "-" << right << endl;
			//		mask2.at<uchar>(Point(y, x)) = 0;
			//	}

			//}


				//So the XoYoR color space stands for " XoYoR, where “o” stands for
				/*logical OR, which combines the X and Y color channels from the XYZ
					color space with the R color channel from the RGB color space
					So I need to have two different items to compare*/
			cvtColor(origimg, Converted, COLOR_BGR2XYZ);
			Mat extract = Converted.clone();
			cout << mask2.rows << " " << mask2.cols << " " << extract.rows << " " << extract.cols << endl;
			for (int x = 0; x < Converted.cols; x++)
			{
				for (int y = 0; y < Converted.rows; y++)
				{
					//AT WORKED WHEN PUTTING IT IN A EMPTY MAT UNLIKE DATA! This will confuse me till the end of time as I have no clue what i sleepily did on tuesday
					if (mask2.at<uchar>(Point(x, y)))
					{
						extract.at<Vec3b>(Point(x, y)) = Vec3f(255, 255, 255);
					}
				}
			}
			if (CTR <= 3) { //Comment this if manual
				//Calculate measurements

				float TP = 0; // TRUE Positives
				float TN = 0;
				float FP = 0; // FALSE POSITIVES
				float FN = 0;
				auto black = Vec3b(255, 255, 255);
				
				for (int x = 0; x < GT.cols; x++)
				{
					for (int y = 0; y < GT.rows; y++)
					{
						//correct Background
						if (GT.at<uchar>(Point(x, y)) == uchar(255))
						{
							//Backgrounds line up
							if (extract.at<Vec3b>(Point(x, y)) == black)
							{
								++TN;
							}
							else
							{
								//false positive
								++FP;
							}

						}
						//Correct Extract
						if (GT.at<uchar>(Point(x, y)) != uchar(255)) {
							if (extract.at<Vec3b>(Point(x, y)) != black)
							{
								//Correct extract
								++TP;
							}
							else {
								//FALSE Background
								++FN;
							}
						}
					}
				}

				float Sensitivity = (TP / (TP + FN)) * 100.0;
				float Specificity = (TN / (TN + FN)) * 100.0;
				float Accuracy = ((TP + TN) / (TP + FP + FN + TN)) * 100.0;
				float Similarity = ((2 * TP) / (2 * TP + FN + FP)) * 100.0;
				std::cout << int(GT.at<uchar>(Point(0, 0))) << extract.at<Vec3b>(Point(0, 0)) << (GT.at<uchar>(Point(0, 0)) == uchar(225)) << (extract.at<Vec3b>(Point(0, 0))) << endl;
				std::cout << "---------Comparison against extract----------" << endl;
				std::cout << "TN: " << TN << " TP: " << TP << " FP: " << FP << " FN: " << FN << endl;
				std::cout << "Sensitivity: " << Sensitivity << endl;
				std::cout << "Specificity: " << Specificity << endl;
				std::cout << "Accuracy: " << Accuracy << endl;
				std::cout << "Similarity: " << Similarity << endl;
				TP = TN = FP = FN = 0;

				for (int x = 0; x < GT.cols; x++)
				{
					for (int y = 0; y < GT.rows; y++)
					{
						//correct Background
						if (GT.at<uchar>(Point(x, y)) == uchar(255))
						{
							if (mask2.at<uchar>(Point(x, y)) == 255)
							{
								++TN;
							}
							else
							{
								//false positive
								++FP;
							}

						}
						//Correct Extract
						if (GT.at<uchar>(Point(x, y)) != uchar(255)) {
							if (mask2.at<uchar>(Point(x, y)) != 255)
							{
								//Correct extract
								++TP;
							}
							else {
								//FALSE Background
								++FN;
							}
						}

					}
				}

				Sensitivity = (TP / (TP + FN)) * 100.0;
				Specificity = (TN / (TN + FN)) * 100.0;
				Accuracy = ((TP + TN) / (TP + FP + FN + TN)) * 100.0;
				Similarity = ((2 * TP) / (2 * TP + FN + FP)) * 100.0;
				std::cout << "---------Comparison against MASK2----------" << endl;
				std::cout << "TN: " << TN << " TP: " << TP << " FP: " << FP << " FN: " << FN;
				std::cout << "Sensitivity: " << Sensitivity << endl;
				std::cout << "Specificity: " << Specificity << endl;
				std::cout << "Accuracy: " << Accuracy << endl;
				std::cout << "Similarity: " << Similarity << endl;
			} //Comment if manual or trying item with no ground truth
			Mat greyex;
			cv::cvtColor(extract, greyex, COLOR_XYZ2BGR);
			cv::cvtColor(greyex, greyex, COLOR_BGR2GRAY);


			//Attempt to diagnose
			///Cant really do circles becuase circles are far from perferct and very far from imperfect
			//CAN do parallel ridge pattern detection.
			cv::cvtColor(extract, extract, COLOR_XYZ2BGR);
			extract = origimg.clone();
			std::cout << mask2.rows << " " << mask2.cols << " " << extract.rows << " " << extract.cols << endl;
			for (int x = 0; x < Converted.cols; x++)
			{
				for (int y = 0; y < Converted.rows; y++)
				{
					//AT WORKED WHEN PUTTING IT IN A EMPTY MAT UNLIKE DATA! This will confuse me till the end of time as I have no clue what i sleepily did on tuesday
					if (mask2.at<uchar>(Point(x, y)))
					{
						extract.at<Vec3b>(Point(x, y)) = Vec3f(255, 255, 255);
					}
				}
			}


			//For a rough throw together NOT BAD

			float ridges = 0;
			float lesionarea = 0;
			float brown = 0;
			float black_ = 0;
			Vec3b brwn = (92, 64, 51);
			for (int x = 0; x < extract.cols; x += 2)
			{
				for (int y = 0; y < extract.rows; y++)
				{
					if (extract.at<Vec3b>(Point(x, y)) != Vec3b(255, 255, 255)) {
						lesionarea++;
					}
					if (x - 1 >= 0 && extract.at<Vec3b>(Point(x, y)) != Vec3b(255, 255, 255) && extract.at<Vec3b>(Point(x - 1, y)) != Vec3b(255, 255, 255)) {
						float dist = pow(extract.at<Vec3b>(Point(x, y))[0] - extract.at<Vec3b>(Point(x - 1, y))[0], 2) + pow(extract.at<Vec3b>(Point(x, y))[1] - extract.at<Vec3b>(Point(x - 1, y))[1], 2) + pow(extract.at<Vec3b>(Point(x, y))[2] - extract.at<Vec3b>(Point(x - 1, y))[2], 2);
						dist = sqrt(dist);
						//cout << dist << " " << extract.at<Vec3b>(Point(x, y)) << " " << extract.at<Vec3b>(Point(x - 1, y)) << endl;
						//Distance needs to be VERY LOW because of bluring
						if (dist >= 6) {
							extract.at<Vec3b>(Point(x, y)) = Vec3b(255, 0, 0);
							++ridges;
						}
						//Brown Color is called "wood bark" BGR<6,10,35>
						dist = pow(extract.at<Vec3b>(Point(x, y))[0] - 6, 2) + pow(extract.at<Vec3b>(Point(x, y))[1] - 10, 2) + pow(extract.at<Vec3b>(Point(x, y))[2] - 35, 2);
						dist = sqrt(dist);
						if (dist <= 30) {
							++brown;
						}
						dist = pow(extract.at<Vec3b>(Point(x, y))[0] - 0, 2) + pow(extract.at<Vec3b>(Point(x, y))[1] - 0, 2) + pow(extract.at<Vec3b>(Point(x, y))[2] - 0, 2);
						dist = sqrt(dist);
						if (dist <= 35) {
							++black_;
						}
					}
				}
			}

			//Auto Clustering
			int i = 0;
			Mat clustSearch = mask2.clone();
			for (int x = 0; x < clustSearch.cols; x++)
			{
				for (int y = 0; y < clustSearch.rows; y++)
				{
					if (clustSearch.at<uchar>(Point(x, y)) == 0) {
						floodFill(clustSearch, Point(x, y), Scalar(i + 50));
						++i;
					}
				}
			}



			namedWindow("clustSearch", WINDOW_AUTOSIZE);
			cv::imshow("clustSearch", clustSearch);
			cv::moveWindow("clustSearch", 0, 90);
			int count = 0;

			std::cout << "Area of Legion:" << lesionarea << endl;
			std::cout << "clusters: " << i << endl;
			std::cout << ridges << endl;
			std::cout << "precentage that are ridges: " << (ridges / lesionarea) * 100.0 << endl;
			if (((ridges / lesionarea) * 100.0) > 15) {
				std::cout << "There is parallel-ridges pressent" << endl;
				++count;
			}
			if ((brown + black_ / lesionarea) * 100.0 >= 40) {
				cout << "more than 40% of legions are blown or black" << endl;
				++count;
			}

			if (i < 4) {
				cout << "made of few clusters" << endl;
				++count;
			}

			cout << " The extracted leagion fulfills " << count << "/3 of the detectable melanoma characteristics" << endl;
			if (count <= 1) {
				cout << "The extracted legion is CONFIDENTLY not melanoma" << endl;
			}
			if (count == 2) {
				cout << "The extracted legion is MOST LIKELY melanoma" << endl;
			}
			if (count == 3) {
				cout << "The extracted legion is CONFIDENTLY melanoma" << endl;
			}

			cout << "residents of b town: " << (brown + black_) << " " << ((brown + black_) / lesionarea) * 100.0 << endl;
			namedWindow("Orginial", WINDOW_AUTOSIZE);
			cv::imshow("Original", origimg);
			cv::moveWindow("Original", 0, 90);



			/*namedWindow("Mask", WINDOW_AUTOSIZE);
			cv::imshow("Mask", mask);
			cv::moveWindow("Mask", 0, 90);*/

			/*namedWindow("inpaint", WINDOW_AUTOSIZE);
			cv::imshow("inpaint", inpaint);
			cv::moveWindow("inpaint", 0, 90);*/

			namedWindow("Converted", WINDOW_AUTOSIZE);
			cv::imshow("Converted", Converted);
			cv::moveWindow("Converted", 0, 90);

			namedWindow("Mask2", WINDOW_AUTOSIZE);
			cv::imshow("Mask2", mask2);
			cv::moveWindow("Mask2", 0, 90);

			namedWindow("equalize", WINDOW_AUTOSIZE);
			cv::imshow("equalize", img);
			cv::moveWindow("equalize", 0, 90);

			namedWindow("GT", WINDOW_AUTOSIZE);
			cv::imshow("GT", GT);
			cv::moveWindow("GT", 0, 90);
			cvtColor(mask2, mask2, COLOR_GRAY2BGR);
			namedWindow("extract", WINDOW_AUTOSIZE);
			cv::imshow("extract", extract);
			cv::moveWindow("extract", 0, 90);
			/*imwrite("../OpenCVImages/Inpaints/Converted" + to_string(CTR) + ".png", Converted);
			imwrite("../OpenCVImages/Inpaints/mask2_" + to_string(CTR) + ".png", mask2);
			imwrite("../OpenCVImages/Inpaints/equalized" + to_string(CTR) + ".png", img);
			imwrite("../OpenCVImages/Inpaints/Extract" + to_string(CTR) + ".png", extract);
			imwrite("../OpenCVImages/Inpaints/flood" + to_string(CTR) + ".png", flood);*/
			//Melanoma Diagnosis
			++CTR;
			cv::waitKey(0);
			//Go in and see if I can diagnose if something is melanoma or not based off of visual symtoms recognized
		} //Remove this in manual

			cv::waitKey(0);
			cv::destroyAllWindows();
			return 0;
}

