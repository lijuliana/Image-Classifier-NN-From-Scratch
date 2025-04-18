import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.util.Scanner;
import javax.imageio.ImageIO;

public class ProcessGray {

    public static final String BIN_INPUT = ".bin";
    public static final int IMG_HEIGHT = 150;
    public static final int IMG_WIDTH = 100;

    public static int[][] create2DIntMatrixFromFile(String inputFile, int width, int height) throws Exception
    {
        int[][] matrix = new int[width][height];
        InputStream inputStream = new FileInputStream(inputFile);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                int val = (int) inputStream.read();
                if (val != -1)
                    matrix[i][j] = val;
            }
        }

        inputStream.close();
        return matrix;
    }

    public static void write2Binary(PelArray pixels, String outFile) throws Exception
    {
        int[][] mat = pixels.getPelArray();
        FileOutputStream fOutStream = new FileOutputStream(outFile);
        DataOutputStream out = new DataOutputStream(fOutStream);

        for (int i = 0; i < mat.length; i++)
        {
            for (int j = 0; j < mat[0].length; j++)
            {
                out.writeByte((byte) mat[i][j]);
            }
        }
    }

    public static void main(String[] args) throws Exception
    {
        PelArray pixels = new PelArray();
        PelArray pixelsProcess;
        String fileName;
        String[] imageFilesList = {
            // "1_1.bin","1_2.bin","1_3.bin","1_4.bin","1_5.bin",
            // "2_1.bin","2_2.bin","2_3.bin","2_4.bin","2_5.bin",
            // "3_1.bin","3_2.bin","3_3.bin","3_4.bin","3_5.bin",
            // "4_1.bin","4_2.bin","4_3.bin","4_4.bin","4_5.bin",
            "5_1.bin","5_2.bin","5_3.bin","5_4.bin","5_5.bin",
            // "6_1.bin","6_2.bin","6_3.bin","6_4.bin","6_5.bin"
        };//Image2GrayBin.listFiles(BIN_INPUT);

        int xCom, yCom;

        for (String fileNameExt : imageFilesList)
        {
            fileName = Image2GrayBin.IMG_DIR + fileNameExt.substring(0, fileNameExt.length() - BIN_INPUT.length());

            pixels.setPelArray(create2DIntMatrixFromFile(Image2GrayBin.IMG_DIR + fileNameExt, IMG_WIDTH, IMG_HEIGHT));
            pixelsProcess = pixels;
            // pixelsProcess = pixelsProcess.grayScaleImage();
            pixelsProcess = pixelsProcess.forceMax(170, PelArray.WHITE);
            pixelsProcess = pixelsProcess.onesComplimentImage();
            // pixelsProcess = pixelsProcess.offsetColors(0,0,-50); // ?
            // pixelsProcess = pixelsProcess.forceMin(500, PelArray.BLACK);
            // write2Binary(pixelsProcess, fileName + "_beforeCOM" + BIN_INPUT);

            // xCom = pixelsProcess.getXcom();
            // yCom = pixelsProcess.getYcom();
            // System.out.println(xCom + ", " + yCom);
            // pixelsProcess = pixelsProcess.crop(Math.max(0, xCom - 900), Math.max(0, yCom - 1350), Math.min(1800, xCom + 900), Math.min(2700, yCom + 1350));
            // pixelsProcess = pixelsProcess.scale(150, 100);
            write2Binary(pixelsProcess, fileName + "_processed" + BIN_INPUT);
            System.out.printf("Finished %s processing\n", fileName);

            String fileNameExt2 = fileNameExt.substring(0, fileNameExt.length() - ProcessGray.BIN_INPUT.length()) + "_processed.bin";
            String fileName2 = Image2GrayBin.IMG_DIR + fileNameExt2.substring(0, fileNameExt2.length() - ProcessGray.BIN_INPUT.length());
            String[] argsToBGR2BMP = new String[] {"gray", Integer.toString(150), Integer.toString(100), Image2GrayBin.IMG_DIR + fileNameExt2, fileName2 + ".bmp"};
            BGR2BMP.main(argsToBGR2BMP);
        }
    }
}
