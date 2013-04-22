using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace OpenCVTut
{
    public partial class Form1 : Form
    {
        private Capture cap;
        private HaarCascade haarface, haarlefteye,haarrighteye;
        string HaarPath;
        
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            // passing 0 gets zeroth webcam
            if (radioButton3.Checked)
                cap = new Capture(@"G:\Documents and Settings\Divey Gupta\Desktop\Divey's Input.avi");
            else if (radioButton2.Checked)
                cap = new Capture(@"G:\Documents and Settings\Divey Gupta\Desktop\Divey's Input.avi");
            else
                cap = new Capture(0);


            // adjust path to find your xml
            HaarPath = @"D:\DC\Softwares\OpenCV\libemgucv-windows-x86-gpu-2.2.1.1150\opencv\data\haarcascades\";
            haarface = new HaarCascade(HaarPath + "haarcascade_frontalface_alt_tree.xml");
            haarlefteye = new HaarCascade(HaarPath + "haarcascade_mcs_lefteye.xml");
            haarrighteye = new HaarCascade(HaarPath + "haarcascade_mcs_righteye.xml");
            
            splitContainer1.SplitterDistance = cap.Width;

            Timer t1 = new Timer();
            t1.Interval = 200;
            t1.Tick += new EventHandler(t1_Tick);
            t1.Start();
        }

        private void t1_Tick(object sender, EventArgs e)
        {
            // nextFrame has the image captured from the source, it is displayed in PictureBox1
            using (Image<Bgr, byte> nextFrame = cap.QueryFrame())
            {
                if (nextFrame != null)
                {
                    // there's only one channel (greyscale), hence the zero index
                    Image<Gray, byte> grayframe = nextFrame.Convert<Gray, byte>();
                    
                    var faces = grayframe.DetectHaarCascade(
                                    haarface, 1.4, 4,
                                    HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                                    new Size(nextFrame.Width / 8, nextFrame.Height / 8)
                                    )[0];


                    //pictureBox1.Image = nextFrame.ToBitmap();
                    Rectangle org = nextFrame.ROI;
                    
                    
                    //foreach (var face in faces)
                    if(faces.Count() != 0)
                    {
                        var face = faces[0];
                        nextFrame.Draw(face.rect, new Bgr(0, double.MaxValue, 0), 3);
                        
                        // Select Left portion of the face
                        nextFrame.ROI = new Rectangle(face.rect.X,face.rect.Y,face.rect.Width/2,face.rect.Height);
                        
                        PictureBoxLeftFace.Image = nextFrame.ToBitmap();
                        Image<Gray, byte> halfgrayface = nextFrame.Convert<Gray, byte>();
                        var eyes = halfgrayface.DetectHaarCascade(
                                    haarlefteye, 1.4, 4,
                                    HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                                    new Size(nextFrame.Width / 8, nextFrame.Height / 8)
                                    )[0];

                        //find smallest eye
                        if (eyes.Count() != 0)
                        {
                            var smallestEye = eyes[0];
                            foreach (var eye in eyes)
                            {
                                if (smallestEye.rect.Width > eye.rect.Width)
                                    smallestEye = eye;
                            }

                            nextFrame.Draw(smallestEye.rect, new Bgr(0, double.MaxValue, 0), 1);
                      
                            nextFrame.ROI = new Rectangle(face.rect.X + smallestEye.rect.X, face.rect.Y + smallestEye.rect.Y, smallestEye.rect.Width, smallestEye.rect.Height);
                      
                            Image<Xyz, byte> xyzLefteyeframe = nextFrame.Convert<Xyz, byte>();
                            Image<Gray, byte>[] xyzLefteyeframes = xyzLefteyeframe.Split();

                            //---- For Eye Bounds
                            Image<Gray, byte> LeftEyeBoundsFrame = xyzLefteyeframes[0].Copy();

                            // For Eye Bounds ----- ///



                            System.IntPtr srcPtr = xyzLefteyeframes[0];
                            System.IntPtr dstPtr = xyzLefteyeframes[0];
                            CvInvoke.cvThreshold(srcPtr, dstPtr, trackBarLeftPupil.Value,255, THRESH.CV_THRESH_BINARY);
                            pictureBoxLeftPupil.Image = xyzLefteyeframes[0].ToBitmap();

                            
                            // To Find Pupil
                            xyzLefteyeframes[0] = xyzLefteyeframes[0].Canny(new Gray(50), new Gray(0)); //// to be checked
                            var contours = xyzLefteyeframes[0].FindContours();

                            Rectangle pupilRect = new Rectangle();
                            while(contours != null)
                            {
                                if (contours.BoundingRectangle.Width * contours.BoundingRectangle.Height > pupilRect.Size.Height * pupilRect.Size.Width)
                                {
                                    if (contours.BoundingRectangle.Y > smallestEye.rect.Height / 10.0) //To Eliminate EyeBrows detected as pupil
                                            pupilRect = contours.BoundingRectangle;
                                }
                                contours = contours.HNext;
                            }

                            nextFrame.Draw(pupilRect, new Bgr(Color.Green), 1);
                            //pictureBox7.Image = nextFrame.ToBitmap();
                            
                            //----- For EyeBounds
                                                                                  
                            Rectangle EyeBoundsRect = new Rectangle();
                            int top = smallestEye.rect.Height, bottom = 0, left = smallestEye.rect.Width, right = 0;
                            //pictureBox12.Image = EyeBoundsFrame.ToBitmap();

                            LeftEyeBoundsFrame = LeftEyeBoundsFrame.ThresholdToZero(new Gray(trackBarLeftPupil.Value));
                            for (int a = 0; a < LeftEyeBoundsFrame.Height; a++)
                            {
                                for (int b = 0; b < LeftEyeBoundsFrame.Width; b++)
                                {
                                    if (new Gray(0).Equals(LeftEyeBoundsFrame[a, b]))
                                        LeftEyeBoundsFrame[a, b] = new Gray(255);
                                }
                            }
                            pictureBoxLeftEyeBoundThresholding.Image = LeftEyeBoundsFrame.ToBitmap();

                            LeftEyeBoundsFrame = LeftEyeBoundsFrame.ThresholdBinary(new Gray(trackBarLeftEyeBound.Value), new Gray(255));
                            pictureBoxLeftEyeBoundsCanny.Image = LeftEyeBoundsFrame.ToBitmap();

                            LeftEyeBoundsFrame = LeftEyeBoundsFrame.Canny(new Gray(50), new Gray(0)); //// to be checked
                            var EyeBoundscontours = LeftEyeBoundsFrame.FindContours();

                            while (EyeBoundscontours != null)
                            {
                                if (EyeBoundscontours.BoundingRectangle.Y > smallestEye.rect.Height / 10)
                                {
                                    if (EyeBoundscontours.BoundingRectangle.Y + EyeBoundscontours.BoundingRectangle.Height > pupilRect.Y)
                                    {
                                        if (top > EyeBoundscontours.BoundingRectangle.Top)
                                            top = EyeBoundscontours.BoundingRectangle.Top;
                                        if (left > EyeBoundscontours.BoundingRectangle.Left)
                                            left = EyeBoundscontours.BoundingRectangle.Left;
                                        if (right < EyeBoundscontours.BoundingRectangle.Right)
                                            right = EyeBoundscontours.BoundingRectangle.Right;
                                        if (bottom < EyeBoundscontours.BoundingRectangle.Bottom)
                                            bottom = EyeBoundscontours.BoundingRectangle.Bottom;
                                    }
                                }
                                EyeBoundscontours = EyeBoundscontours.HNext;
                            }

                            EyeBoundsRect.X = left;
                            EyeBoundsRect.Y = top;
                            EyeBoundsRect.Height = bottom - top;
                            EyeBoundsRect.Width = right - left;
                            nextFrame.Draw(EyeBoundsRect, new Bgr(Color.Blue), 1);

                            Image<Gray, byte> leftErode = LeftEyeBoundsFrame.Copy();
                            Image<Gray,byte> leftDilate = LeftEyeBoundsFrame.Copy();
                            leftErode._Erode(trackBarErodeDilate.Value);
                            pictureBoxLeftEyeErode.Image = leftErode.ToBitmap();
                            leftDilate._Dilate(trackBarErodeDilate.Value);
                            pictureBoxLeftEyeDilate.Image = leftDilate.ToBitmap();



                            // For Eye Bounds -------////
                           
                       }


                        
                        ////----- Right Portion -----/////
                        // Select Right portion of the face
                        nextFrame.ROI = new Rectangle(face.rect.X + face.rect.Width / 2, face.rect.Y, face.rect.Width / 2, face.rect.Height);
                        
                        pictureBoxRightFace.Image = nextFrame.ToBitmap();
                        halfgrayface = nextFrame.Convert<Gray, byte>();
                        eyes = halfgrayface.DetectHaarCascade(
                                    haarrighteye, 1.4, 4,
                                    HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                                    new Size(nextFrame.Width / 8, nextFrame.Height / 8)
                                    )[0];

                        if (eyes.Count() != 0)
                        {
                            var smallestEye = eyes[0];
                            foreach (var eye in eyes)
                            {
                                if (smallestEye.rect.Width > eye.rect.Width)
                                    smallestEye = eye;
                            }

                            nextFrame.Draw(smallestEye.rect, new Bgr(0, double.MaxValue, 0), 1);
                           
                            nextFrame.ROI = new Rectangle(face.rect.X + face.rect.Width / 2 + smallestEye.rect.X, face.rect.Y + smallestEye.rect.Y, smallestEye.rect.Width, smallestEye.rect.Height);
                            

                            Image<Xyz, byte> xyzrighteyeframe = nextFrame.Convert<Xyz, byte>();
                            Image<Gray, byte>[] xyzrighteyeframes = xyzrighteyeframe.Split();

                            //---- For Eye Bounds
                            Image<Gray, byte> RightEyeBoundsFrame = xyzrighteyeframes[0].Copy();
                            // For Eye Bounds ---- //

                            Image<Gray, byte> adaptivePupil = xyzrighteyeframes[0].Clone();
                            System.IntPtr srcPtr = xyzrighteyeframes[0];
                            System.IntPtr dstPtr = xyzrighteyeframes[0]; 
                            CvInvoke.cvThreshold(srcPtr, dstPtr, trackBarRightPupil.Value,255, THRESH.CV_THRESH_BINARY);
                            pictureBoxRightPupil.Image = xyzrighteyeframes[0].ToBitmap();
                            
                            
                            //To Find Pupil
                            xyzrighteyeframes[0] = xyzrighteyeframes[0].Canny(new Gray(50), new Gray(0));
                            var contours = xyzrighteyeframes[0].FindContours();

                            Rectangle pupilRect = new Rectangle();
                            while (contours != null)
                            {
                                if (contours.BoundingRectangle.Width * contours.BoundingRectangle.Height > pupilRect.Size.Height * pupilRect.Size.Width)
                                {
                                    if (contours.BoundingRectangle.Y > smallestEye.rect.Height / 10.0) //To Eliminate EyeBrows detected as pupil
                                        pupilRect = contours.BoundingRectangle;
                                }
                                contours = contours.HNext;
                            }

                            nextFrame.Draw(pupilRect, new Bgr(Color.Green), 1);
                            //pictureBox11.Image = nextFrame.ToBitmap();

                            //----- For EyeBounds

                            Rectangle EyeBoundsRect = new Rectangle();
                            int top = smallestEye.rect.Height, bottom = 0, left = smallestEye.rect.Width, right = 0;
                            //pictureBox12.Image = EyeBoundsFrame.ToBitmap();

                            RightEyeBoundsFrame = RightEyeBoundsFrame.ThresholdToZero(new Gray(trackBarRightPupil.Value));
                            for (int a = 0; a < RightEyeBoundsFrame.Height; a++)
                            {
                                for (int b = 0; b < RightEyeBoundsFrame.Width; b++)
                                {
                                    if (new Gray(0).Equals(RightEyeBoundsFrame[a, b]))
                                        RightEyeBoundsFrame[a, b] = new Gray(255);
                                }
                            }
                            pictureBoxRightEyeBoundThresholding.Image = RightEyeBoundsFrame.ToBitmap();

                            RightEyeBoundsFrame =RightEyeBoundsFrame.ThresholdBinary(new Gray(trackBarRightEyeBound.Value), new Gray(255));
                            pictureBoxRightEyeBoundsCanny.Image = RightEyeBoundsFrame.ToBitmap();

                            RightEyeBoundsFrame = RightEyeBoundsFrame.Canny(new Gray(50), new Gray(0)); //// to be checked
                            var EyeBoundscontours = RightEyeBoundsFrame.FindContours();

                            while (EyeBoundscontours != null)
                            {
                                if (EyeBoundscontours.BoundingRectangle.Y > smallestEye.rect.Height / 10)
                                {
                                    if (EyeBoundscontours.BoundingRectangle.Y + EyeBoundscontours.BoundingRectangle.Height > pupilRect.Y)
                                    {
                                        if (top > EyeBoundscontours.BoundingRectangle.Top)
                                            top = EyeBoundscontours.BoundingRectangle.Top;
                                        if (left > EyeBoundscontours.BoundingRectangle.Left)
                                            left = EyeBoundscontours.BoundingRectangle.Left;
                                        if (right < EyeBoundscontours.BoundingRectangle.Right)
                                            right = EyeBoundscontours.BoundingRectangle.Right;
                                        if (bottom < EyeBoundscontours.BoundingRectangle.Bottom)
                                            bottom = EyeBoundscontours.BoundingRectangle.Bottom;
                                    }
                                }
                                EyeBoundscontours = EyeBoundscontours.HNext;
                            }

                            EyeBoundsRect.X = left;
                            EyeBoundsRect.Y = top;
                            EyeBoundsRect.Height = bottom - top;
                            EyeBoundsRect.Width = right - left;
                            nextFrame.Draw(EyeBoundsRect, new Bgr(Color.Blue), 1);

                            Image<Gray, byte> rightErode = RightEyeBoundsFrame.Copy();
                            Image<Gray, byte> rightDilate = RightEyeBoundsFrame.Copy();
                            rightErode._Erode(trackBarErodeDilate.Value);
                            pictureBoxRightEyeErode.Image = rightErode.ToBitmap();
                            rightDilate._Dilate(trackBarErodeDilate.Value);
                            pictureBoxRightEyeDilate.Image = rightDilate.ToBitmap();
                            
                            // For Eye Bounds -------////

                            
                            if(radioButton4.Checked)
                                Cursor.Position = new Point(smallestEye.rect.X * 10, smallestEye.rect.Y * 10);
                           
                            
                        }

                        


                    }
                    nextFrame.ROI = org;
                    pictureBox1.Image = nextFrame.ToBitmap();
                    

                }
            }
        }

        private void pictureBox3_Click(object sender, EventArgs e)
        {

        }

        private void radioButton2_Click(object sender, EventArgs e)
        {
            cap = new Capture(@"G:\Documents and Settings\Divey Gupta\Desktop\Divey's Input.avi");

            trackBarLeftPupil.Value = 29;
            trackBarLeftEyeBound.Value = 51;
            trackBarRightPupil.Value = 33;
            trackBarRightEyeBound.Value = 56;

        }

        private void radioButton3_Click(object sender, EventArgs e)
        {
            cap = new Capture(@"G:\Documents and Settings\Divey Gupta\Desktop\Divey's Input.avi");

            trackBarLeftPupil.Value = 19;
            trackBarLeftEyeBound.Value = 34;
            trackBarRightPupil.Value = 19;
            trackBarRightEyeBound.Value = 35;
        }

        private void radioButton1_Click(object sender, EventArgs e)
        {
            cap = new Capture(0);
        }

        private void trackBarLeftPupil_ValueChanged(object sender, EventArgs e)
        {
            labelLeftPupil.Text = trackBarLeftPupil.Value.ToString();
        }

        private void trackBarLeftEyeBound_ValueChanged(object sender, EventArgs e)
        {
            labelLeftEyeBound.Text = trackBarLeftEyeBound.Value.ToString();
        }

        private void trackBarRightPupil_ValueChanged(object sender, EventArgs e)
        {
            labelRightPupil.Text = trackBarRightPupil.Value.ToString();
        }

        private void trackBarRightEyeBound_ValueChanged(object sender, EventArgs e)
        {
            labelRightEyeBound.Text = trackBarRightEyeBound.Value.ToString();
        }

        private void trackBarErodeDilate_ValueChanged(object sender, EventArgs e)
        {
            labelErodeDilate.Text = trackBarErodeDilate.Value.ToString();
        }

        
    }
}
