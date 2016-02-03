import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Muffff on 1/31/16.
 * ****************** TESTING THE VERSION CONTROL!!!!
 */
public class Controller {

    List<Mat> featureList;

    HOGDescriptor hog;
    /**
     * HOG Parameters
     */
    Size windowSize;
    Size cellSize;
    Size blockSize;
    Size winStride;
    Size padding;
    Size blockStride;
    int nBins;


    public Controller() {
        initializeHOG();
    }


    /**
     * Get all JPG file names from given directory.
     *
     * @param dir path to directory
     * @return list of all PNG file names
     */

    public List<String> getFilenames(String dir) throws IOException {
        List<String> fileNames = new ArrayList<>();

        File[] files = new File(dir).listFiles();

        String filePath = null;

        System.out.println("Directory: " + dir);

        if (files != null) {

            for (File file : files) {
                if (file.isFile() && file.getName().toLowerCase().endsWith(".jpg")) {
                    filePath = file.getCanonicalPath();
                    fileNames.add(filePath);
                }

            }
        } else System.out.println("Empty directory");

        return fileNames;
    }

    public Mat createModel(String dir) throws IOException {

        //32 bit Floating Point with 1 channel
        Mat data = new Mat(0, 0, CvType.CV_32FC1);

        Mat labels = new Mat(0, 0, CvType.CV_32FC1);

//        featureList = new ArrayList<>();

        MatOfFloat descriptor = new MatOfFloat();

        MatOfPoint locations = new MatOfPoint();

        List<String> listOfFiles = getFilenames(dir);

        for (String file : listOfFiles) {

            if (file != null) {
                Mat img_raw = Imgcodecs.imread(file);
                if (img_raw.empty()) {
                    System.out.println("Image loaded is null");
                    break;
                }

                Mat img_gray = launchPreProcessor(img_raw);
                hog.compute(img_gray, descriptor, winStride, padding, locations);
                float label = Float.parseFloat(file.substring(38, 39));
                Mat l = new Mat(1, 1, CvType.CV_32FC1, new Scalar(label));
                labels.push_back(l);

                data.push_back(descriptor.clone());

                System.out.println("** done ** \n" + file + " has been computed");
            } else
                System.out.println("NULL FILE");
        }

        return data;

    }

    protected Mat launchPreProcessor(Mat img_raw) {
        //640,480
        Size imageResize = new Size(128, 64);

        Mat img_gray = new Mat();

        Mat img_resized = new Mat();

        Imgproc.resize(img_raw, img_resized, imageResize);

        Imgproc.cvtColor(img_raw, img_gray, Imgproc.COLOR_BGR2GRAY);


        return img_gray;
//        displayImage(img_gray);
    }


    protected void initializeHOG() {

        windowSize = new Size(128, 64);

        cellSize = new Size(8, 8);

        blockSize = new Size(16, 16);

        winStride = new Size(windowSize.width / 2, windowSize.height / 2);

        padding = new Size(0, 0);

        //50% overlap
        blockStride = new Size(blockSize.width / 2, blockSize.height / 2);

        nBins = 9;

        hog = new HOGDescriptor(windowSize, blockSize, blockStride, cellSize, nBins);

    }

//    protected void computeHOG()
//    {
//
//        //HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins)
//
//        Mat gradients = new Mat();
//
//        //READ UPON PUSH_BACK
//        gradients.push_back(descriptor.clone());
//
//        String saveLoc = "/Users/Muffff/Development/desc/hogdesc.xml";
//
//        hog.save(saveLoc);
//
////        write(descriptor);
//
//    }
}
