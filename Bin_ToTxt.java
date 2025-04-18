import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.FileWriter;

public class Bin_ToTxt {
   public static void main(String[] args) throws IOException {
      FileWriter myWriter = new FileWriter("Image_Train.txt");
      for (int imgNum = 1; imgNum <= 5; imgNum++) {
         for (int imgType = 1; imgType <= 6; imgType++) {
            if (imgType == 5) continue;
            Path path = Paths.get("./Processed_Bin/" + imgType + "_" + imgNum + "_processed.bin");
            byte[] fileContents =  Files.readAllBytes(path);

            for (byte b : fileContents)
               // if (b == 0) myWriter.write(0.0 + " "); else myWriter.write(1.0 + " ");
               myWriter.write((b & 0xff)/256.0 + " ");
            myWriter.write("\n");
         }
      }
      myWriter.close();

      FileWriter myWriter2 = new FileWriter("Image_Test.txt");
      for (int imgNum = 1; imgNum <= 5; imgNum++) {
         for (int imgType = 5; imgType <= 5; imgType++) {
            Path path = Paths.get("./Processed_Bin/" + imgType + "_" + imgNum + "_processed.bin");
            byte[] fileContents =  Files.readAllBytes(path);

            for (byte b : fileContents)
               // if (b == 0) myWriter2.write(0.0 + " "); else myWriter2.write(1.0 + " ");
               myWriter2.write((b & 0xff)/256.0 + " ");
            myWriter2.write("\n");
         }
      }
      myWriter2.close();
   }

}