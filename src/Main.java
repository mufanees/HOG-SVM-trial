import org.opencv.core.Core;

import java.io.IOException;

/**
 * Created by Muffff on 1/31/16.
 */
public class Main {

    public static void main(String[] args) throws IOException {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

//        launch(args);


        Controller c1 = new Controller();

        c1.createModel("/Users/Muffff/Development/Dataset/pos/");
//
//        String str = "/Users/Muffff/Development/Dataset/pos/10000File.JPG";
//        System.out.println(str.substring(38,39));

    }
}
