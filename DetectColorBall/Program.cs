using System;
using System.Drawing;
using System.IO;
using System.Numerics;
using System.Reflection;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Ocl;
using Emgu.CV.Util;

namespace DetectColorBall
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

        static void Main(string[] args)
        {
            var outputDir = AppDomain.CurrentDomain.BaseDirectory;

            var crochetPath = Path.GetFullPath(Path.Combine(outputDir, @"Images\\crochet.jpg"));
            var crochetMat = new Emgu.CV.Mat(crochetPath);
            var crochetImg = crochetMat.ToImage<Bgr, byte>();

            var rainbowPath = Path.GetFullPath(Path.Combine(outputDir, @"Images\\rainbow.jpg"));
            var rainbowMat = new Emgu.CV.Mat(rainbowPath);
            var rainbowImg = rainbowMat.ToImage<Bgr, byte>();
            
            CopyToImage(ref crochetImg, ref rainbowImg, 100, 100);
            //CvInvoke.Imshow(rainbowImg.ToString(), rainbowImg);
            
            /// Exercise 4 ///
            var crochetGrayImg = new Image<Bgr, byte>(crochetPath); ;
            CvInvoke.CvtColor(crochetImg, crochetGrayImg, ColorConversion.Bgr2Gray);
            //CvInvoke.Imshow(crochetGrayImg.ToString(), crochetGrayImg);
            
            var crochetFlipImg = new Image<Bgr, byte>(crochetPath);
            CvInvoke.Flip(crochetFlipImg, crochetFlipImg, FlipType.Vertical);
            //CvInvoke.Imshow(crochetFlipImg.ToString(), crochetFlipImg);

            var finalCrochet = new Emgu.CV.Mat(
                (crochetImg.Height + crochetGrayImg.Height + crochetFlipImg.Height) / 3,
                crochetImg.Width + crochetGrayImg.Width + crochetFlipImg.Width, DepthType.Cv32F, 3);
            var finalCrochetImg = finalCrochet.ToImage<Bgr, byte>();

            CopyToImage(ref crochetImg, ref finalCrochetImg);
            GrayCopyChannelToImage(ref crochetGrayImg, ref finalCrochetImg, crochetGrayImg.Width);
            CopyToImage(ref crochetFlipImg, ref finalCrochetImg, crochetFlipImg.Width * 2);
            //CvInvoke.Imshow(finalCrochet.ToString(), finalCrochetImg);

            /// Exercise 5 ///
            var crochetHsvImg = new Image<Hsv, byte>(crochetPath);
            CvInvoke.CvtColor(crochetImg, crochetHsvImg, ColorConversion.Bgr2Hsv);
            //CvInvoke.Imshow(crochetHsvImg.ToString(), crochetHsvImg);
            
            var crochetFinalHImg = new Image<Hsv, byte>(crochetPath);
            var crochetFinalSImg = new Image<Hsv, byte>(crochetPath);
            var crochetFinalVImg = new Image<Hsv, byte>(crochetPath);
            GrayCopyChannelToImage(ref crochetHsvImg, ref crochetFinalHImg, 0, 0, 0);
            //CvInvoke.Imshow(crochetFinalHImg.ToString(), crochetFinalHImg);
            GrayCopyChannelToImage(ref crochetHsvImg, ref crochetFinalSImg, 0, 0, 1); 
            //CvInvoke.Imshow(crochetFinalSImg.ToString(), crochetFinalSImg);
            GrayCopyChannelToImage(ref crochetHsvImg, ref crochetFinalVImg, 0, 0, 2);
            //CvInvoke.Imshow(crochetFinalVImg.ToString(), crochetFinalVImg);

            var hsvPath = Path.GetFullPath(Path.Combine(outputDir, @"Images\\hsv.png"));
            Hsv hsvMin = new Hsv(50, 0, 0);
            Hsv hsvMax = new Hsv(75, 255, 255);
            
            var crochetHsvSeuilImg = new Image<Hsv, byte>(crochetPath);
            CvInvoke.InRange(crochetHsvImg, new ScalarArray(hsvMin.MCvScalar), new ScalarArray(hsvMax.MCvScalar), crochetHsvSeuilImg);
            //CvInvoke.Imshow(crochetHsvSeuilImg.ToString(), crochetHsvSeuilImg);

            /// Exercise 6 ///
            
            var kernelOp = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(11, 11), new Point(-1, -1));
            
            CvInvoke.MorphologyEx(crochetHsvSeuilImg, crochetHsvSeuilImg, 
                MorphOp.Open, kernelOp, new Point(-1, -1), 1,
                BorderType.Default, new MCvScalar());
            CvInvoke.Imshow(crochetHsvSeuilImg.ToString(), crochetHsvSeuilImg);

            /// Exercise 7 ///
            
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(crochetHsvSeuilImg, contours, hierarchy, 
                RetrType.List, ChainApproxMethod.ChainApproxSimple);
            //CvInvoke.ContourArea(contours);

            int largestContourIndex = -1;
            double largestContour = double.NegativeInfinity;
            for (int i = 0; i < contours.Size; i++)
            {
                double contour = CvInvoke.ContourArea(contours[i]);
                if (largestContour < contour)
                {
                    largestContour = contour;
                    largestContourIndex = i;
                }
            }

            if (largestContourIndex >= 0)
            {
                CvInvoke.DrawContours(crochetHsvSeuilImg, contours, largestContourIndex, new MCvScalar(255, 0, 0), 2);
            }

            CvInvoke.WaitKey();
        }
    }
}
