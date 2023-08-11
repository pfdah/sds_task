package test;
import java.io.File;

public class TestSolution {
	private static final String CSV_1_PATH = "../resources/1.csv";
	private static final String CSV_2_PATH = "../resources/2.csv";
	private static final String OUTPUT_CSV_PATH = "../resources/my_output.csv";

	public void combineFiles(File csv1, File csv2, File outputFile) throws Exception {
       	System.out.println(csv1.);
	}

	public static void main( String[] args ) {

		try {

			File csv1 = new File( CSV_1_PATH );
			File csv2 = new File( CSV_2_PATH );
			File output = new File( OUTPUT_CSV_PATH );

			TestSolution solution = new TestSolution( );
			solution.combineFiles( csv1, csv2, output );

		} catch ( Exception e ) {
			e.printStackTrace( );
		}
	}
}
