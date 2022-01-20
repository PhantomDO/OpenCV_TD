using System;
using System.IO;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Ocl;
using Emgu.CV.Util;
using Emgu.CV.Aruco;
using Emgu.CV.VideoStab;

namespace QRCode
{
    class Program
    {
        public static void CopyToImage<TColor1, TColor2, TDepth>(
            ref Image<TColor1, TDepth> a, ref Image<TColor2, TDepth> b,
            int offsetX = 0, int offsetY = 0)
            where TColor1 : struct, IColor
            where TColor2 : struct, IColor
            where TDepth : new()
        {
            var aData = a.Data;
            var bData = b.Data;
            for (uint y = 0; y < aData.GetLength(0); y++)
            {
                for (uint x = 0; x < aData.GetLength(1); x++)
                {
                    for (uint z = 0; z < aData.GetLength(2); z++)
                    {
                        if (y + offsetY < bData.GetLength(0) && x + offsetX < bData.GetLength(1))
                        {
                            bData[y + offsetY, x + offsetX, z] = aData[y, x, z];
                        }
                        //else
                        //{
                        //    Console.WriteLine($"Over Size Data : ({y + offsetY} => {bData.GetLength(0)}), ({x + offsetX} => {bData.GetLength(1)})");
                        //}
                    }
                }
            }

            b.Data = bData;
        }
        
        public static void GrayCopyChannelToImage<TColor1, TColor2, TDepth>(
            ref Image<TColor1, TDepth> a, ref Image<TColor2, TDepth> b,
            int offsetX = 0, int offsetY = 0, int canalIndex = 0)
            where TColor1 : struct, IColor
            where TColor2 : struct, IColor
            where TDepth : new()
        {
            var aData = a.Data;
            var bData = b.Data;
            for (uint y = 0; y < aData.GetLength(0); y++)
            {
                for (uint x = 0; x < aData.GetLength(1); x++)
                {
                    for (uint z = 0; z < aData.GetLength(2); z++)
                    {
                        if (y + offsetY < bData.GetLength(0) && x + offsetX < bData.GetLength(1))
                        {
                            bData[y + offsetY, x + offsetX, z] = aData[y, x, canalIndex];
                        }
                        //else
                        //{
                        //    Console.WriteLine($"Over Size Data : ({y + offsetY} => {bData.GetLength(0)}), ({x + offsetX} => {bData.GetLength(1)})");
                        //}
                    }
                }
            }

            b.Data = bData;
        }

        static string GetImagePath(string imgName)
        {
            var outputDir = AppDomain.CurrentDomain.BaseDirectory;
            var folderPath = Path.Combine(outputDir, @"images\\");
            return Path.GetFullPath(Path.Combine(folderPath, imgName));
        }

        static void GetMarkers(ref Mat mat)
        {
            VectorOfInt ids = new VectorOfInt();
            VectorOfVectorOfPointF acceptedCorners = new VectorOfVectorOfPointF();
            VectorOfVectorOfPointF ignoreCorners = new VectorOfVectorOfPointF();
            DetectorParameters detectorParameters = new DetectorParameters();
            detectorParameters = DetectorParameters.GetDefault();

            Dictionary dictMarkers = new Dictionary(Dictionary.PredefinedDictionaryName.Dict6X6_250);
            Mat grayFrame = new Mat(mat.Width, mat.Height, DepthType.Cv8U, 1);
            CvInvoke.CvtColor(mat, grayFrame, ColorConversion.Bgr2Gray);
            ArucoInvoke.DetectMarkers(mat, dictMarkers, acceptedCorners, ids, detectorParameters, ignoreCorners);

            Mat display = new Mat(mat.Width, mat.Height, DepthType.Cv8U, 3);
            mat.CopyTo(display);

            if (ids.Size > 0)
            {
                ArucoInvoke.DrawDetectedMarkers(display, acceptedCorners, ids, new MCvScalar(0,255,0));
            }

            CvInvoke.Imshow(display.ToString(), display);
        }

        static void CreateMarker(ref Mat mat)
        {
            Mat grayFrame = new Mat(mat.Width, mat.Height, DepthType.Cv8U, 1);
            CvInvoke.CvtColor(mat, grayFrame, ColorConversion.Bgr2Gray);

            Mat thresholdFrame = new Mat(mat.Width, mat.Height, DepthType.Cv8U, 1); ;

            CvInvoke.AdaptiveThreshold(grayFrame, thresholdFrame, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 15, 20);

            Image<Bgr, byte> displayImage = new Image<Bgr, byte>(mat.Width * 2, mat.Height);
            Image<Bgr, byte> frameImage = mat.ToImage<Bgr, byte>();
            Image<Gray, byte> thresholdImage = thresholdFrame.ToImage<Gray, byte>();

            CopyToImage(ref frameImage, ref displayImage);
            CopyToImage(ref thresholdImage, ref displayImage, frameImage.Width);

            CvInvoke.Imshow(displayImage.ToString(), displayImage);
        }

        static void Main(string[] args)
        {
            var imgAruco = new Mat(GetImagePath("aruco.png"));
            var imgAruco_d = new Mat(GetImagePath("aruco_d.png"));
            var imgAruco_d2 = new Mat(GetImagePath("aruco_d2.png"));
            var imgGrumpy = new Mat(GetImagePath("grumpy.png"));
            var imgMarkersInTheWild = new Mat(GetImagePath("markersInTheWild.jpg"));

            //GetMarkers(ref imgAruco);
            CreateMarker(ref imgAruco);
            CvInvoke.WaitKey();
        }
    }
}
