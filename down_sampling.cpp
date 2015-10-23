#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

using namespace cv;

/// Global variables
Mat src, dst, tmp;
string window_name = "Pyramids Demo";

/**
 * @function main
 */
int main( int argc, char** argv )
{
    string readfile;
    double scaling_factor;
    bool automatic_flag = true;
    if ( argc < 2 )
    {
        readfile = "Original_images/obj_001_001.jpg";
    }
    else 
    {
         readfile = argv[1];
         if (argc > 2)
         {
             scaling_factor = strtod(argv[2],NULL);
             std::cout<< "the scaling factor is : "<< scaling_factor<<"\n";
         }
         else {
             scaling_factor = 0.8;
         }
    }
  /// General instructions
  printf( "\n Zoom In-Out demo  \n " );
  printf( "------------------ \n" );
  printf( " * [u] -> Zoom in  \n" );
  printf( " * [d] -> Zoom out \n" );
  printf( " * [p] -> Save image \n" );
  printf( " * [s] -> blur \n" );
  printf( " * [ESC] -> Close program \n \n" );

  /// Test image - Make sure it s divisible by 2^{n}
  src = imread( readfile );
  if( !src.data )
    { printf(" No data! -- Exiting the program \n");
      return -1; }

  tmp = src;
  dst = tmp;

  /// Create window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );
  imshow( window_name, dst );

  /// Loop
  int i = 0;
  int j = 0;
  while( true )
  {
    int c;
    c = waitKey(10);

    if( (char)c == 27 )
      { break; }
    if( (char)c == 'u' )
      { pyrUp( tmp, dst, Size( tmp.cols*2, tmp.rows*2 ) );
        printf( "** Zoom In: Image x 2 \n" );
        i--;
      }
    else if( (char)c == 'd' )
     { 
       resize( tmp, dst, Size( tmp.cols*scaling_factor, tmp.rows*scaling_factor ) );
       printf( "** Zoom Out: Image \n" );
       i++;
     }
     else if( (char)c == 's' )
     { 
       GaussianBlur(tmp, dst, Size(5,5) , 0);
       printf( "** Blured : Image \n" );
       j++;
     }
     else if( (char)c == 'p' )
     { 
        string write_name ;
        std::stringstream sstm;
        sstm << "img_smoth_"<<j;
        sstm << "down_"<<i<<".jpg";
        write_name = sstm.str();
        std::cout<< write_name <"\n";
        imwrite(write_name, dst);
     }

    imshow( window_name, dst );
    tmp = dst;
  }
  return 0;
}
