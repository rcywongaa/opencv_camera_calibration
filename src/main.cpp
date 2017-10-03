#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"
#include <signal.h>
#include <unistd.h>

using namespace std;
using namespace cv;

const float square_size_m = 0.01905;
const Size num_intersections = Size(6, 9); //horizontal intersections, vertical intersections

bool isContinue = true;

void sigint_handler(int s){
    printf("--- SIGINT ---\n");
    isContinue = false;
}

vector<Point3f> createIntersectionPositions(Size intersection_count, float square_size)
{
    vector<Point3f> intersection_positions;
    for (int i = 0; i < intersection_count.height; i++)
    {
        for (int j = 0; j < intersection_count.width; j++)
        {
            intersection_positions.push_back(Point3f(j * square_size, i * square_size, 0.0f));
        }
    }
    return intersection_positions;
}



int main(int argc, char** argv)
{
    signal(SIGINT, sigint_handler);
    VideoCapture cap(0); // open the default camera
    while(isContinue && !cap.isOpened())  // check if we succeeded
    {
        printf("Failed to open camera, retrying...\n");
        sleep(1);
        cap.open(0);
    }

    vector<Mat> found_chessboards;
    vector<vector<Point2f>> found_corners_per_mat;

    while(isContinue)
    {
        Mat frame;
        Mat corners_frame;
        cap >> frame; // get a new frame from camera
        vector<Point2f> corners;
        bool is_found = findChessboardCorners(frame, num_intersections, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
        if (is_found)
        {
            //found_chessboards.push_back(frame);
            found_corners_per_mat.push_back(corners);
            printf("num of samples: %d\n", found_corners_per_mat.size());
        }
        frame.copyTo(corners_frame);
        drawChessboardCorners(corners_frame, num_intersections, corners, is_found);
        imshow("found corners", corners_frame);
        if (is_found) waitKey(1000);
        else waitKey(1);
    }

    printf("Capture finished, processing...\n");

    if (found_corners_per_mat.size() > 0)
    {
        vector<vector<Point3f>> true_intersections_per_mat;
        true_intersections_per_mat.push_back(createIntersectionPositions(num_intersections, square_size_m));
        true_intersections_per_mat.resize(found_corners_per_mat.size(), true_intersections_per_mat[0]);

        vector<Mat> rVectors, tVectors;
        Mat distance_coefficients = Mat::zeros(8, 1, CV_64F);
        Mat camera_matrix = Mat::eye(3, 3, CV_64F);

        double rms = calibrateCamera(true_intersections_per_mat, found_corners_per_mat, num_intersections, camera_matrix, distance_coefficients, rVectors, tVectors);

        cout << "Camera matrix = " << endl << camera_matrix << endl;
        printf("rms error = %f\n", rms);
        printf("Focal length (in pixels) = (%f, %f)\n", camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1));
        printf("Optical center (in pixels) = (%f, %f)\n", camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 2));
    }
    else printf("No calibration corners found\n");

    return 0;
}
