import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ScaleBMP {

   public static BufferedImage convertToBufferedImage(Image img) {

      if (img instanceof BufferedImage) {
         return (BufferedImage) img;
      }

      // Create a buffered image with transparency
      BufferedImage bi = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

      Graphics2D graphics2D = bi.createGraphics();
      graphics2D.drawImage(img, 0, 0, null);
      graphics2D.dispose();

      return bi;
   }

   public static void main(String[] args) throws IOException {
      for (int imgType = 5; imgType <= 5; imgType++) {
         for (int imgNum = 2; imgNum <= 5; imgNum++) {
            BufferedImage bmp = ImageIO.read(new File("./BMP_Images/" + imgType + "_" + imgNum + ".bmp"));

            int w = 150;
            int h = 100;

            Image scaledIMG = bmp.getScaledInstance(w, h, Image.SCALE_SMOOTH);
            
            BufferedImage scaledBMP = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

            scaledBMP.getGraphics().drawImage(scaledIMG, 0, 0, null);

            File scaledBMPFile = new File("./BMP_Images/" + imgType + "_" + imgNum + "_scaled.bmp");
            ImageIO.write(scaledBMP, "bmp", scaledBMPFile);
         }
      }
   }

}