import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
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

    SVM classifier;

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
        initialize();
    }


    /**
     * Get all JPG file names from given directory.
     *
     * @param dir path to directory
     * @return list of all PNG file names
     */

    public List<String> getFilenames(String dir) throws IOException {

        System.out.println("Directory: " + dir);

        List<String> fileNames = new ArrayList<>();

        File[] files = new File(dir).listFiles();

        for (File file : files) {
            if (file.isFile() && file.getName().toLowerCase().endsWith(".jpg")) {
                fileNames.add(file.getName());
            }

        }

        return fileNames;
    }

    public void createModel(String dir) {

        //32 bit Floating Point with 1 channel
        Mat data = new Mat(0, 0, CvType.CV_32FC1);
        //32 bit Integer with 1 channel
        Mat labels = new Mat(0, 0, CvType.CV_32SC1);

//      featureList = new ArrayList<>();

        List<String> listOfFiles = null;

        try {
            listOfFiles = getFilenames(dir);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (String file : listOfFiles) {

            if (file != null) {
                Mat img_raw = Imgcodecs.imread(dir + file);
                if (img_raw.empty()) {
                    System.out.println("Image loaded is null");
                    continue;
                }

                Mat img_processed = launchPreProcessor(img_raw);
                data.push_back(computeHOG(img_processed));

                int label = Integer.parseInt(file.substring(0, 1));
                Mat l = new Mat(1, 1, CvType.CV_32SC1, new Scalar(label));
                labels.push_back(l);

                System.out.println("\nImage size: " + img_processed.size());
                System.out.println("** done ** \n" + file + " has been computed, label: " + label);

            } else
                System.out.println("NULL FILE");
        }

        System.out.println("\nData Size: " + data.size() + "\nLabels Size: " + labels.size());

        classifier.train(data, Ml.ROW_SAMPLE, labels);
        System.out.println(classifier.isTrained());

//        classifier.save("lampSVMTrained.xml");

        testTrainedModel("/Users/Muffff/Development/Dataset/test/");

    }

    protected Mat computeHOG(Mat img_processed) {

        MatOfFloat descriptor = new MatOfFloat();
        MatOfPoint locations = new MatOfPoint();

        hog.compute(img_processed, descriptor, winStride, padding, locations);

        return descriptor.reshape(0, 1);

    }

    protected void testTrainedModel(String dir) {

        System.out.println(classifier);

        List<String> listOfFiles = null;

        try {
            listOfFiles = getFilenames(dir);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (String file : listOfFiles) {

            if (file != null) {

                Mat img_raw = Imgcodecs.imread(dir + file);

                if (img_raw.empty()) {
                    System.out.println("Image loaded is null");
                    continue;
                }

                Mat img_processed = launchPreProcessor(img_raw);
                System.out.println("Label for image " + file + ": " + classifier.predict(computeHOG(img_processed)));

            }
        }


    }

    protected Mat launchPreProcessor(Mat img_raw) {
        //640,480
        Size imageResize = new Size(128, 64);

        Mat img_gray = new Mat();

        Mat img_resized = new Mat();

        Imgproc.cvtColor(img_raw, img_gray, Imgproc.COLOR_BGR2GRAY);

        Imgproc.resize(img_gray, img_resized, imageResize);

        return img_resized;
//        displayImage(img_gray);
    }


    protected void initialize() {

        windowSize = new Size(128, 64);
        cellSize = new Size(8, 8);
        blockSize = new Size(16, 16);
        winStride = new Size(windowSize.width / 2, windowSize.height / 2);
        padding = new Size(0, 0);
        //50% overlap
        blockStride = new Size(blockSize.width / 2, blockSize.height / 2);
        nBins = 9;

        hog = new HOGDescriptor(windowSize, blockSize, blockStride, cellSize, nBins);

        classifier = SVM.create();

        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 100, 0.1);
        classifier.setKernel(SVM.LINEAR);
        classifier.setC(1);
        classifier.setTermCriteria(criteria);
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
